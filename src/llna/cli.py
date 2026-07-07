"""Command-line interface.

    llna rules --resolution 9 --cache-dir data/rule_tables
    llna sweep --network-type small-world --resolution 5 --output results.h5

`llna sweep` is the batched, seeded, resumable successor of the historical
scripts/state_vs-defect-density.py: for every non-equivalent rule it evolves
an ensemble of initial configurations together with single-defect twins and
stores the pooled convergence statistics of the state and defect densities.
Output schema matches the historical results_*.h5 files:
resolution{R}/{network_type}/{state|defect}/{median|q1|q3}.
"""

import argparse
import logging
import random
import sys

import igraph as ig
import numpy as np
import torch as tc

from llna.analysis import init_config_with_dens, median_and_percentiles_over_ensemble
from llna.io import SweepFile
from llna.networks import create_2d_torus_lattice, watts_strogatz_rewire
from llna.rules import get_nonequiv_rules, load_nonequiv_rules
from llna.sweep import auto_rule_chunk, evolve_defect_pairs

log = logging.getLogger("llna")

NETWORK_TYPES = ("toroidal-lattice", "small-world", "random")


def edge_tensor(graph: ig.Graph) -> tc.Tensor:
    g = graph.copy()
    g.to_directed()
    return tc.tensor(g.get_edgelist(), dtype=tc.long).T


def build_graphs(network_type: str, num_graphs: int, L: int, degree: int, rewiring_prob: float) -> list:
    """Seeding note: igraph and watts_strogatz_rewire draw from the global
    `random` module; the caller seeds it once from --seed."""
    if network_type == "toroidal-lattice":
        return [create_2d_torus_lattice(L, degree=degree)] * num_graphs
    if network_type == "small-world":
        base = create_2d_torus_lattice(L, degree=degree)
        return [watts_strogatz_rewire(base, rewiring_prob) for _ in range(num_graphs)]
    if network_type == "random":
        n, m = L * L, L * L * degree // 2
        graphs = []
        while len(graphs) < num_graphs:
            candidate = ig.Graph.Erdos_Renyi(n=n, m=m)
            if candidate.is_connected():
                graphs.append(candidate)
        return graphs
    raise ValueError(f"Unknown network type {network_type!r}; choose from {NETWORK_TYPES}.")


def make_ensemble(
    rng: np.random.Generator, num_graphs: int, num_init_conf: int, num_nodes: int, init_dens: float
):
    """Per graph: [L, N] configurations with exact density init_dens, plus
    single-defect twins. Defect positions are unique across the whole
    ensemble (the historical variance-reduction choice), which requires
    num_graphs * num_init_conf <= num_nodes."""
    total = num_graphs * num_init_conf
    if total > num_nodes:
        raise ValueError(
            f"num_graphs*num_init_conf ({total}) exceeds the number of nodes ({num_nodes}); "
            "unique single-defect positions are impossible."
        )
    configs = np.stack([init_config_with_dens(num_nodes, init_dens, rng=rng) for _ in range(total)]).reshape(
        num_graphs, num_init_conf, num_nodes
    )
    defect_nodes = rng.choice(num_nodes, size=total, replace=False).reshape(num_graphs, num_init_conf)
    defected = configs.copy()
    for g in range(num_graphs):
        defected[g, np.arange(num_init_conf), defect_nodes[g]] ^= 1
    return configs, defected


def cmd_rules(args) -> int:
    table = load_nonequiv_rules(args.resolution, cache_dir=args.cache_dir)
    expected = (4**args.resolution + 2**args.resolution) // 2
    location = (
        f"{args.cache_dir}/all_nonequiv_res{args.resolution}_rules.npy" if args.cache_dir else "(not cached)"
    )
    print(
        f"resolution {args.resolution}: {len(table)} non-equivalent rules (expected {expected}) -> {location}"
    )
    return 0


def cmd_sweep(args) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    num_nodes = args.L * args.L
    rules = np.asarray(get_nonequiv_rules(args.resolution), dtype=np.int64)
    graphs = build_graphs(args.network_type, args.num_graphs, args.L, args.degree, args.rewiring_prob)
    edges = [edge_tensor(g) for g in graphs]
    configs, defected = make_ensemble(rng, args.num_graphs, args.num_init_conf, num_nodes, args.init_dens)
    log.info(
        "sweep: %d rules, %d graphs (%s), %d configs/graph, N=%d, T=%d, device=%s",
        len(rules),
        args.num_graphs,
        args.network_type,
        args.num_init_conf,
        num_nodes,
        args.T,
        args.device,
    )

    writer = SweepFile(args.output)
    group = f"resolution{args.resolution}/{args.network_type}"
    columns = {
        f"{kind}/{stat}": ((), np.float64) for kind in ("state", "defect") for stat in ("median", "q1", "q3")
    }
    attrs = {k: v for k, v in vars(args).items() if k != "func"}
    done = writer.init_group(group, rules, columns, attrs)
    if done:
        log.info("resuming at rule %d/%d", done, len(rules))

    chunk = args.rule_chunk or auto_rule_chunk(len(rules), 2 * args.num_init_conf, num_nodes)
    for start in range(done, len(rules), chunk):
        rule_chunk = rules[start : start + chunk]
        state_tails, defect_tails = [], []
        for E, graph_configs, graph_defected in zip(edges, configs, defected, strict=True):
            states, defects = evolve_defect_pairs(
                E,
                graph_configs,
                graph_defected,
                rule_chunk,
                args.resolution,
                iso=True,
                T=args.T,
                device=args.device,
                rule_chunk=len(rule_chunk),
            )
            state_tails.append(states[:, :, -args.delta_t :])
            defect_tails.append(defects[:, :, -args.delta_t :])
        # pool graphs x configs x final delta_t timesteps per rule
        state_tail = np.concatenate(state_tails, axis=1)
        defect_tail = np.concatenate(defect_tails, axis=1)
        rows = {name: np.empty(len(rule_chunk)) for name in columns}
        for i in range(len(rule_chunk)):
            m, q1, q3 = median_and_percentiles_over_ensemble(state_tail[i], args.delta_t, time_axis=1)
            rows["state/median"][i], rows["state/q1"][i], rows["state/q3"][i] = m, q1, q3
            m, q1, q3 = median_and_percentiles_over_ensemble(defect_tail[i], args.delta_t, time_axis=1)
            rows["defect/median"][i], rows["defect/q1"][i], rows["defect/q3"][i] = m, q1, q3
        writer.write_rows(group, start, rows)
        log.info("rules %d-%d/%d done", start, min(start + chunk, len(rules)), len(rules))
    log.info("sweep complete -> %s:%s", args.output, group)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="llna", description="Life-Like Network Automata toolkit.")
    sub = parser.add_subparsers(required=True)

    p_rules = sub.add_parser("rules", help="Enumerate non-equivalent rules (optionally cache to disk).")
    p_rules.add_argument("--resolution", type=int, required=True)
    p_rules.add_argument("--cache-dir", default=None, help="e.g. data/rule_tables")
    p_rules.set_defaults(func=cmd_rules)

    p_sweep = sub.add_parser("sweep", help="State/defect convergence sweep over all non-equivalent rules.")
    p_sweep.add_argument("--network-type", choices=NETWORK_TYPES, required=True)
    p_sweep.add_argument("--resolution", type=int, default=5, help="Odd (isomorphic encoding).")
    p_sweep.add_argument("--L", type=int, default=30, help="Grid side; N = L*L nodes.")
    p_sweep.add_argument("--degree", type=int, default=8)
    p_sweep.add_argument("--rewiring-prob", type=float, default=0.2)
    p_sweep.add_argument("--num-graphs", type=int, default=30)
    p_sweep.add_argument("--num-init-conf", type=int, default=30)
    p_sweep.add_argument("--init-dens", type=float, default=0.5)
    p_sweep.add_argument("--T", type=int, default=100)
    p_sweep.add_argument("--delta-t", type=int, default=10, help="Final timesteps pooled for the statistics.")
    p_sweep.add_argument("--seed", type=int, default=42)
    p_sweep.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    p_sweep.add_argument("--rule-chunk", type=int, default=None)
    p_sweep.add_argument("--output", default="results.h5")
    p_sweep.set_defaults(func=cmd_sweep)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
