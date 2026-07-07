"""Derrida maps: analytical, empirical, coefficients, and spline crossings."""

from collections.abc import Sequence

import igraph as ig
import numpy as np
import torch as tc
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from scipy.optimize import root_scalar
from scipy.stats import binom, hypergeom

from llna.analysis._encoding import _interval_encoding
from llna.analysis.ensemble import _random_ones_arrays
from llna.automata import LLNA


def derrida_map_analytical(
    resolution: int,
    B_set: NDArray[np.int_] | Sequence[int],
    S_set: NDArray[np.int_] | Sequence[int],
    delta0: float | NDArray,
    degree: int,
    init_config_dens: float = 0.5,
    iso: bool = True,
) -> NDArray:
    """
    Calculate the analytical Derrida map for a defect of normalised Hamming weight d0, from the local update rule for a particular node degree.
    There is no need for a LLNA object (so the calculation is generally faster).
    NOTE: this defaults to iso=True right now.
    TODO: this is very slow and should be massively optimised.

    Parameters
    ----------
    resolution : int
        Positive integer indicating the resolution of the local update rule.
    B_set : list or numpy.ndarray
        list of integers corresponding to the indices of the activated density intervals in the B set.
    S_set : list or numpy.ndarray
        list of integers corresponding to the indices of the activated density intervals in the B set.
    degree : int
        The degree of the node you want to calculate the Derrida map for.
    delta0 : float
        The normalised Hamming weight of the defect. Value between 0 and 1.
    init_config_dens : float
        The density of the initial configuration (before defect). Default is 0.5.
    iso : bool
        True by default. Set to False if non-isomorphic density intervals are used in the LLNA definition.

    Returns
    -------
    delta1 : float
        The Derrida map for this particular degree. Value between 0 and 1.
    """
    # check that the input values make sense
    if (degree < 1) or (degree > 1022):
        raise ValueError(
            f"Degree {degree} not allowed. The degree must be a non-zero natural number smaller than 1023."
        )
    if iso and resolution % 2 == 0:
        raise ValueError(
            f"Resolution {resolution} is not possible when iso=True. Choose an odd positive integer."
        )
    if (B_set and (np.max(B_set) >= resolution)) or (S_set and (np.max(S_set) >= resolution)):
        raise ValueError(f"Resolution {resolution} is to small for the provided update intervals.")
    delta0 = np.atleast_1d(delta0)
    if (np.min(delta0) < 0) or (np.max(delta0) > 1):
        raise ValueError(
            f"Normalised Hamming weight delta0 must be between 0 and 1. Now the maximum is {np.max(delta0)} and the minimum is {np.max(delta0)}."
        )
    if (init_config_dens < 0) or (init_config_dens > 1):
        raise ValueError(f"Density rho must be between 0 and 1, not {init_config_dens}.")

    # define the big XOR function that is at the end of all the summations
    def function_xor(
        resolution,  # r
        B_set,  # B
        S_set,  # S
        centre_state,  # s
        num_living,  # q
        centre_toggle,  # c
        num_killer_toggles,  # t
        num_defects,  # d
        degree,  # k
        iso=True,  # isomorphic interval encoding
    ):
        # but B_set and S_set in a list
        B_and_S_set = [B_set, S_set]
        # calculate the density interval for the default and the defect case
        rho = num_living / degree
        rho_defect = (num_living - 2 * num_killer_toggles + num_defects) / degree
        rho_interval = _interval_encoding(resolution, np.array([[rho]]), iso=iso)[0].argmax()
        rho_defect_interval = _interval_encoding(resolution, np.array([[rho_defect]]), iso=iso)[0].argmax()
        # calculate the default function response
        B_or_S_set = B_and_S_set[centre_state]
        default_output = int(rho_interval in B_or_S_set)
        # calculate the defect function response
        centre_defect = (centre_state + centre_toggle) % 2
        B_or_S_set_defect = B_and_S_set[centre_defect]
        defect_output = int(rho_defect_interval in B_or_S_set_defect)
        # calculate whether they are the same and output the XOR
        xor_output = int(default_output != defect_output)
        return xor_output

    # define all relevant probabilities
    def prob_centre_node_alive(centre_state, rho0):
        return rho0**centre_state * (1 - rho0) ** (1 - centre_state)

    def prob_q_living_neighbours(num_living, degree, rho0):
        return binom.pmf(num_living, degree, rho0)

    def prob_centre_node_toggle(centre_toggle, delta0):
        return delta0**centre_toggle * (1 - delta0) ** (1 - centre_toggle)

    def prob_d_toggles(num_defects, degree, delta0):
        return binom.pmf(num_defects, degree, delta0)

    def prob_killer_toggle(num_killer_toggles, degree, num_living, num_defects):
        return hypergeom.pmf(num_killer_toggles, degree, num_living, num_defects)

    # sum over all the possible combinations
    # TODO: this is coded pretty badly with all these loops ...
    delta1 = np.zeros_like(delta0)
    for centre_state in [0, 1]:
        prob1 = prob_centre_node_alive(centre_state, init_config_dens)
        for num_living in range(degree + 1):
            prob2 = prob_q_living_neighbours(num_living, degree, init_config_dens)
            for centre_toggle in [0, 1]:
                prob3 = prob_centre_node_toggle(centre_toggle, delta0)
                for num_defects in range(degree + 1):
                    prob4 = prob_d_toggles(num_defects, degree, delta0)
                    min_t = max(0, num_defects + num_living - degree)
                    max_t = min(num_defects, num_living)
                    for num_killer_toggles in range(min_t, max_t + 1):
                        prob5 = prob_killer_toggle(num_killer_toggles, degree, num_living, num_defects)
                        function_output = function_xor(
                            resolution,
                            B_set,
                            S_set,
                            centre_state,
                            num_living,
                            centre_toggle,
                            num_killer_toggles,
                            num_defects,
                            degree,
                        )
                        value = prob1 * prob2 * prob3 * prob4 * prob5 * function_output
                        delta1 += value
    return np.clip(delta1, 0, 1)


def get_derrida_arrays(
    graph: ig.Graph,
    model: LLNA,
    points_per_rho: int = 1,
    num_init_configs: int | None = None,
    init_configs: NDArray | None = None,
    return_until_dens: float = 1.0,
) -> tuple[NDArray, NDArray]:
    """
    Function that generates the arrays that are required for creating a Derrida plot. It effectively selects `points_per_rho` randomly chosen defects per normalised Hamming distance.
    TODO: add a parameter to choose the number of random initial conditions
    TODO: add the possiblity of adding an array of multiple initial configurations
    TODO: add possiblity of choosing the order in which the nodes should be affected

    Parameters
    ----------
    graph : igraph.Graph
        Network (graph) used as the topology for the automaton. Will be interpreted as an undirected graph.
    model : src.automata.LLNA
        Life-like network automaton (custom class). Envelops the rules that govern the automaton.
    points_per_rho : int
        The desired number of data points per normalised Hamming weights in the input. If None the number will default to 1.
    num_init_configs : int, optional
        Number of randomly chosen initial configurations. If None, the number of randomly chosen initial configuration defaults to 1. Must be None if the init_config kwarg is used.
    init_configs : numpy.ndarray, optional
        The desired initial configuration. Can be a single array, or an array of arrays. If None, a random initial configuration is generated from a uniform distribution.
    return_until_dens : float
        The maximum density of defects for which the Derrida plot arrays are calculated. Default is 1.0 (the full plot).

    Returns
    -------
    input_defect_density : numpy.ndarray
        Array containing the densities of the defects, i.e. the normalised Hamming distance, of the two original initial configurations.
    output_defect_density : numpy.ndarray
        Array containing the densities of the defects of the two configurations in the next time step.
    """
    # find the number of nodes in the network
    N = graph.vcount()

    # generate or verify the initial configuration
    if init_configs is not None:
        if num_init_configs is not None:
            raise ValueError(
                "When manually entering the initial configurations, the kwarg `num_init_config` must be None."
            )
        if init_configs.shape[-1] != N:
            raise ValueError(f"The provided initial configuration(s) should have length {N}.")
        if init_configs.ndim > 2:
            raise ValueError("The provided initial configuration(s) should have either 1 or 2 dimensions.")
        if init_configs.ndim < 2:  # fix dimensions if just a single array is given
            init_configs = init_configs[np.newaxis, :]
        num_init_configs = init_configs.shape[0]
    else:  # make a single random initial configuration
        if num_init_configs is None:
            num_init_configs = 1
        init_configs = np.random.randint(0, 2, size=(num_init_configs, N))

    # get a long array of unique defects
    if (points_per_rho * num_init_configs) > N:
        raise Exception(
            f"The number of data points per normalised hamming weight cannot be larger than or equal to {N}, because there are not that many unique combinations of defect arrays.\nLower the points per density and/or the number of initial configuration, and/or increase the number of nodes in the network."
        )
    # calculate the cutoff value of the defect density
    if not (1 / N <= return_until_dens <= 1.0):
        raise ValueError(f"The kwarg return_until_dens must be a value between {1 / N} and 1.")
    max_ones_per_array = int(N * return_until_dens)
    # take case where max_ones is N, which is not helpful
    max_ones_per_array = min(N - 1, max_ones_per_array)
    # create array of max ones
    max_ones_per_array_range = range(1, max_ones_per_array + 1)
    # create defect array using a helper function
    defects_all = np.vstack(
        [
            _random_ones_arrays(N, points_per_rho * num_init_configs, ones_per_array)
            for ones_per_array in max_ones_per_array_range
        ]
    )

    # copy the initial configurations points_per_rho times
    num_rhos = len(max_ones_per_array_range)
    init_configs_all = np.tile(init_configs, (points_per_rho * num_rhos, 1))
    # add the defects to the initial configurations in ascending order of number of defects
    init_configs_defect_all = (init_configs_all + defects_all) % 2

    # get ID of edges (bidirectional)
    graph.to_directed()
    edges = tc.tensor(graph.get_edgelist()).T
    graph.to_undirected()

    # run the model for a single time step for the various initial conditions
    next_configs = np.array(model.step(edges, tc.tensor(init_configs)), dtype=int)
    # copy the output as many times as required (for points_per_rho)
    next_config_all = np.tile(next_configs, (points_per_rho * num_rhos, 1))
    # run the model for a single time step for all the defected initial conditions
    next_config_defect_all = np.array(model.step(edges, tc.tensor(init_configs_defect_all)), dtype=int)

    # find the normalised Hamming distance between both new configurations
    input_defect_density = np.mean(defects_all, axis=1)
    output_defect_density = np.mean((next_config_all + next_config_defect_all) % 2, axis=1)

    return input_defect_density, output_defect_density


def calculate_derrida_coefficient(
    rho_t_array: NDArray, rho_tplus1_array: NDArray, cutoff_rho_t: float = 0.05
) -> np.floating:
    """
    Calculate the Derrida coefficient for the given arrays of defect fractions at two consecutive time steps.

    Parameters
    ----------
    rho_t_array : np.ndarray
        Array of defect fractions at time step t.
    rho_tplus1_array : np.ndarray
        Array of defect fractions at time step t+1.
    cutoff_rho_t : float, optional
        Cutoff value for rho_t to consider in the calculation, by default 0.05 (1/20th of the full Derrida map).

    Returns
    -------
    derrida_coefficient : float
        Derrida coefficient.
    """
    # Find cutoff value for input density rho_t
    if np.max(rho_t_array) < cutoff_rho_t:
        raise ValueError(
            f"Maximum value of rho_t_array is less than the requested kwarg value `cutoff_rho_t`={cutoff_rho_t}."
        )
    rho_t_array = rho_t_array[rho_t_array <= cutoff_rho_t]
    rho_tplus1_array = rho_tplus1_array[: len(rho_t_array)]
    # calculate slope using the least squares method from this subset of densities
    derrida_coefficient = np.sum(rho_t_array * rho_tplus1_array) / np.sum(rho_t_array * rho_t_array)

    # return Derrida coefficient. NOTE that different sources use different definitions
    return derrida_coefficient


def derrida_spline_and_roots(
    inputs: np.ndarray, outputs: np.ndarray, num_bins: int = 20
) -> tuple[CubicSpline, list[float]]:
    """
    Fit a cubic spline to Derrida plot data and find the roots where the spline intersects the diagonal.

    Parameters
    ----------
    inputs : numpy.ndarray
        Array of input values (x-coordinates) for the Derrida plot.
    outputs : numpy.ndarray
        Array of output values (y-coordinates) for the Derrida plot.
    num_bins : int, optional
        Number of bins to use for binning the data before fitting the spline. Default is 20.

    Returns
    -------
    spline : scipy.interpolate.CubicSpline
        Cubic spline fitted to the binned data.
    roots : list of float
        List of x-values where the spline intersects the diagonal (y = x).

    Notes
    -----
    The function first sorts the input data and bins it into a specified number of bins.
    It then computes the median y-values in each bin and fits a cubic spline to these binned data points.
    The spline is constrained to pass through the origin (0,0). The function then finds the roots of the
    equation spline(x) - x = 0, which correspond to the points where the spline intersects the diagonal.
    """
    # Sort data
    sorted_indices = np.argsort(inputs)
    x_sorted = inputs[sorted_indices]
    y_sorted = outputs[sorted_indices]

    # Bin the data and compute median y-values in each bin
    bins = np.linspace(0, 1, num_bins)
    digitized = np.digitize(x_sorted, bins)
    x_binned = [
        x_sorted[digitized == i].mean() for i in range(1, num_bins) if len(x_sorted[digitized == i]) > 0
    ]
    y_binned = [
        np.median(y_sorted[digitized == i]) for i in range(1, num_bins) if len(y_sorted[digitized == i]) > 0
    ]

    # Ensure the spline passes through the origin by explicitly adding (0,0)
    x_binned.insert(0, np.floating(0))  # Insert x = 0 at the beginning
    y_binned.insert(0, np.floating(0))  # Insert y = 0 at the beginning

    # Fit a cubic spline with natural boundary conditions (does not force a specific slope)
    spline = CubicSpline(x_binned, y_binned, bc_type="natural")  # No clamping, just a natural spline

    # Define function for finding roots: f(x) - x = 0
    def diagonal_crossing(x):
        return spline(x) - x

    # Find all intersection points
    x_fine = np.linspace(0, 1, 500)  # Fine grid for detecting sign changes
    y_diff = spline(x_fine) - x_fine  # Compute f(x) - x

    # Detect sign changes (indicating crossing points)
    roots = []
    for i in range(len(x_fine) - 1):
        if y_diff[i] * y_diff[i + 1] < 0:  # Sign change means a root is between x_fine[i] and x_fine[i+1]
            try:
                root = root_scalar(
                    diagonal_crossing, bracket=[x_fine[i], x_fine[i + 1]], method="brentq"
                ).root
                roots.append(root)
            except ValueError:
                pass  # Skip if no valid root is found
    return spline, roots
