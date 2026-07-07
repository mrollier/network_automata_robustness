# %% LOAD PARSING AND LOGGING

import argparse  # For parsing command-line arguments
import logging   # For logging execution details

# %% LOAD PACKAGES

# standard preamble for the Notebooks I use
import torch as tc
import igraph as ig
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams

# Enable LaTeX and set Times New Roman as the font
rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}"  # Optional: Use LaTeX packages
})

from tqdm import tqdm

# compact saving of data
import h5py

from llna.automata import LLNA
from llna.rules import binary_indices, get_nonequiv_rules
from llna.networks import create_2d_torus_lattice, watts_strogatz_rewire
from llna.analysis import init_config_with_dens, median_and_percentiles_over_ensemble

# %load_ext autoreload
# %autoreload 2

def parse_arguments():
    """Set up and parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Simulate state vs defect density on various network types.")
    parser.add_argument("--L", type=int, default=30, help="Grid size (L x L).")
    parser.add_argument("--rewiring_prob", type=float, default=0.2, help="Rewiring probability for the small-world network.")
    parser.add_argument("--degree", type=int, default=8, help="Degree for lattice model (Moore neighbourhood).")
    parser.add_argument("--num_graphs", type=int, default=30, help="Number of graphs per type.")
    parser.add_argument("--num_init_conf", type=int, default=30, help="Number of initial configurations per graph.")
    parser.add_argument("--init_dens", type=float, default=0.5, help="Initial density of states.")
    parser.add_argument("--T", type=int, default=100, help="Number of time steps for simulation.")
    parser.add_argument("--delta_t", type=int, default=10, help="Interval for calculating percentiles.")
    parser.add_argument("--output_file", type=str, default="state-median-vs-defect-median.h5", help="Output file name for saving results.")
    parser.add_argument("--network_type", type=str, choices=["toroidal-lattice", "small-world", "random"], required=True, help="Type of network to simulate.")
    parser.add_argument("--resolution", type=int, default=5, help="Resolution for the non-equivalent rules.")
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_arguments()

    # Configure logging
    network_type = args.network_type
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"simulation_{network_type}.log")#,
            # logging.StreamHandler()
        ]
    )

    # Log the parsed arguments
    logging.info("Parsed arguments:")
    for arg, value in vars(args).items():
        logging.info(f"  {arg}: {value}")

    # %% CREATE NETWORK

    # PARAMETERS
    #####################################################################################################################
    L = args.L
    num_nodes = L**2
    rewiring_prob = args.rewiring_prob
    degree = args.degree
    num_edges = num_nodes * degree // 2
    num_graphs_per_type = args.num_graphs
    num_init_conf_per_graph = args.num_init_conf
    init_dens = args.init_dens
    T = args.T
    delta_t = args.delta_t
    resolution = args.resolution
    #####################################################################################################################

    logging.info("Simulation parameters initialized.")
    logging.info(f"Grid size: {L}x{L}, Rewiring probability: {rewiring_prob}, Degree: {degree}")
    logging.info(f"Number of graphs: {num_graphs_per_type}, Initial configurations per graph: {num_init_conf_per_graph}")
    logging.info(f"Initial density: {init_dens}, Time steps: {T}, Delta t: {delta_t}")
    logging.info(f"Selected network type: {network_type}")
    logging.info(f"Resolution: {resolution}")

    # Create the selected network type
    if network_type == "toroidal-lattice":
        graph = create_2d_torus_lattice(L, degree=degree)
        graphs = [graph] * num_graphs_per_type
    elif network_type == "small-world":
        base_graph = create_2d_torus_lattice(L, degree=degree)
        graphs = [watts_strogatz_rewire(base_graph, rewiring_prob) for _ in range(num_graphs_per_type)]
    elif network_type == "random":
        graphs = []
        for _ in range(num_graphs_per_type):
            while True:
                random_graph = ig.Graph.Erdos_Renyi(n=num_nodes, m=num_edges)
                if random_graph.is_connected():
                    graphs.append(random_graph)
                    break
    else:
        raise ValueError(f"Invalid network type: {network_type}. Must be one of 'toroidal-lattice', 'small-world', or 'random'.")

    # Find edges
    edges = []
    for graph in graphs:
        graph.to_directed()
        edge = tc.tensor(graph.get_edgelist()).T
        edges.append(edge)
        graph.to_undirected()

    # %% RUN OVER RULES

    beta_sigma_list = get_nonequiv_rules(resolution)

    # open new lists. These will have dimensions [rules]
    state_medians_per_rule = []
    state_q1_per_rule = []
    state_q3_per_rule = []
    defect_medians_per_rule = []
    defect_q1_per_rule = []
    defect_q3_per_rule = []

    logging.info(f"Running through the non-equivalent rules of resolution {resolution} for {network_type} network...")
    # loop over rules
    for i, (beta, sigma) in enumerate(beta_sigma_list):
        logging.info(f"Processed {i}/{len(beta_sigma_list)} rules...")
        # initialise model
        B_set = binary_indices(beta)
        S_set = binary_indices(sigma)
        model = LLNA(resolution, B_set, S_set, iso=True)
        # get initial configurations
        init_configs_array = init_config_with_dens(num_nodes * num_init_conf_per_graph * num_graphs_per_type, init_dens)
        init_configs_array = init_configs_array.reshape(num_graphs_per_type, num_init_conf_per_graph, num_nodes)
        # get initial defects
        init_defects_array = np.eye(num_nodes, dtype=int)[np.random.choice(num_nodes, num_init_conf_per_graph * num_graphs_per_type, replace=False)]
        init_defects_array = init_defects_array.reshape(num_graphs_per_type, num_init_conf_per_graph, num_nodes)
        # find defected initial configurations
        init_configs_with_defect_array = (init_configs_array + init_defects_array) % 2
        # open new arrays that will collect all time series
        state_averages_stack = np.empty((0, T + 1))
        defect_averages_stack = np.empty((0, T + 1))
        # loop over graphs
        for edge, init_configs, init_configs_with_defect in zip(edges, init_configs_array, init_configs_with_defect_array):
            # evolve the configuration
            configs = model.forward(edge, tc.tensor(init_configs), T=T).numpy().astype(int)
            configs_with_defect = model.forward(edge, tc.tensor(init_configs_with_defect), T=T).numpy().astype(int)
            defects = (configs + configs_with_defect) % 2
            # get the average over all nodes
            state_averages = np.mean(configs, axis=2)
            defect_averages = np.mean(defects, axis=2)
            # add these to the stack
            state_averages_stack = np.vstack((state_averages_stack, state_averages))
            defect_averages_stack = np.vstack((defect_averages_stack, defect_averages))
        # get the median and IQR for the ensemble (stacks are [ensemble, T+1])
        state_median, state_q1, state_q3 = median_and_percentiles_over_ensemble(state_averages_stack, delta_t=delta_t, time_axis=1)
        defect_median, defect_q1, defect_q3 = median_and_percentiles_over_ensemble(defect_averages_stack, delta_t=delta_t, time_axis=1)
        # append to lists
        state_medians_per_rule.append(state_median)
        state_q1_per_rule.append(state_q1)
        state_q3_per_rule.append(state_q3)
        defect_medians_per_rule.append(defect_median)
        defect_q1_per_rule.append(defect_q1)
        defect_q3_per_rule.append(defect_q3)

    logging.info(f"Finished processing {network_type} network.")

    # turn lists into numpy arrays
    state_medians_per_rule = np.array(state_medians_per_rule)
    state_q1_per_rule = np.array(state_q1_per_rule)
    state_q3_per_rule = np.array(state_q3_per_rule)
    defect_medians_per_rule = np.array(defect_medians_per_rule)
    defect_q1_per_rule = np.array(defect_q1_per_rule)
    defect_q3_per_rule = np.array(defect_q3_per_rule)

    # %% SAVE FILES

    # Determine file mode based on existence
    file_mode = 'a' if os.path.exists(args.output_file) else 'w'
    logging.info(f"Saving results to {args.output_file} in {'append' if file_mode == 'a' else 'write'} mode...")

    with h5py.File(args.output_file, file_mode) as f:
        group_path = f'resolution{resolution}/{network_type}'
        if group_path in f:
            logging.warning(f"Group {group_path} already exists in the file. Overwriting datasets...")
            del f[group_path]  # Remove existing group to avoid conflicts
        f.create_dataset(f'{group_path}/state/median', data=state_medians_per_rule, compression='gzip')
        f.create_dataset(f'{group_path}/state/q1', data=state_q1_per_rule, compression='gzip')
        f.create_dataset(f'{group_path}/state/q3', data=state_q3_per_rule, compression='gzip')
        f.create_dataset(f'{group_path}/defect/median', data=defect_medians_per_rule, compression='gzip')
        f.create_dataset(f'{group_path}/defect/q1', data=defect_q1_per_rule, compression='gzip')
        f.create_dataset(f'{group_path}/defect/q3', data=defect_q3_per_rule, compression='gzip')

        # Attach metadata if the resolution group is new
        resolution_group = f[f'resolution{resolution}']
        if not resolution_group.attrs:
            resolution_group.attrs['num_nodes'] = f'{L}x{L}'
            resolution_group.attrs['rewiring_prob'] = f'{rewiring_prob}'
            resolution_group.attrs['num_graphs_per_type'] = f'{num_graphs_per_type}'
            resolution_group.attrs['num_init_conf_per_graph'] = f'{num_init_conf_per_graph}'
            resolution_group.attrs['init_dens'] = f'{init_dens}'
            resolution_group.attrs['T'] = f'{T}'
            resolution_group.attrs['delta_t'] = f'{delta_t}'

    logging.info("Results saved successfully.")

if __name__ == "__main__":
    main()