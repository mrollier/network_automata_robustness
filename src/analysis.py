# %% import packages

# regular packages
import numpy as np
import torch as tc
import igraph as ig
from typing import Union, Tuple, Optional, Sequence, List
from scipy.special import comb
from numpy.typing import NDArray
import math

# for spline and root finding
from scipy.interpolate import CubicSpline
from scipy.optimize import root_scalar
from scipy.stats import binom, hypergeom

# warnings and exceptions
import warnings

# particular packages
from src.automata import LLNA
from src.rules import _is_eca
import cellpylib as cpl

# %% utilitary functions
def jacobian(
    graph:ig.Graph,
    model:LLNA,
    states:NDArray[np.int_],
    return_next:bool=False
) -> Union[NDArray[np.int_], Tuple[NDArray[np.int_], NDArray[np.int_]]]:
    """
    Returns the Jacobian matrix of partial Boolean derivatives. J[i,j] can be either 0 or 1,
    and answers the question 'is node i affected by a change in the state of node j in the previous time step?'
    If it is, J[i,j] is 1. Otherwise, J[i,j] is 0.

    Parameters
    ----------
    graph : igraph.Graph
        Network (graph) used as the topology for the automaton. Will be interpreted as a undirected graph.
    model : src.automata.LLNA
        Life-like network automaton (custom class). Envelops the rules that govern the automaton.
    states : numpy.ndarray
        Initial configuration of the LLNA.
    return_next : bool
        Defaults to False. If true, return the configuration in the graph on the next time step (evolved from states via model).

    Returns
    -------
    J : numpy.ndarray
        The Jacobian matrix of size N x N. If changing node j affect node i in the next time step, J[i,j]=1.
    states_next : numpy.ndarray
        The network automaton configuration at the next time step (evolved from 'states').
    """
    ### returns Jacobian matrix with dimensions N x N, where N is the number of nodes in the graph
    # number of nodes
    N = len(states)
    states = states.astype(int)

    # get ID of edges (bidirectional)
    graph.to_directed()
    edges = tc.tensor(graph.get_edgelist()).T
    graph.to_undirected()

    # make matrix with N rows of the same state array
    states_all = np.tile(states, (N,1))
    # calculate unperturbed next-timestep state vector (N times in parallel)
    states_all_next = np.array(model.step(edges, tc.tensor(states_all)), dtype=int)
    states_next = states_all_next[0]

    # make matrix with all possible single-defect perturbations
    states_defect_all = (np.tile(states, (N,1)) + np.diag(np.ones(N, dtype=int))) % 2
    # calculate perturbed next-timestep state vector (N times in parallel)
    states_defect_all_next = np.array(model.step(edges, tc.tensor(states_defect_all)), dtype=int)

    # compile Jacobian with XOR operator (the Boolean derivative)
    J = (states_all_next + states_defect_all_next) % 2

    # return transpose (to match definition)
    if return_next:
        return J.T, states_next
    return J.T

def jacobian_ECA(
    rule:int,
    states:NDArray,
    return_next:bool
) -> Union[NDArray, Tuple[NDArray, NDArray]]:
    """
    TODO: add description
    """
    # number of nodes
    N = len(states)
    # calculate unperturbed next-timestep state vector
    states_next = cpl.evolve(states[np.newaxis,:], timesteps=2, apply_rule=lambda n, c, t: cpl.nks_rule(n, rule))[-1]
    states_all_next = np.tile(states_next, (N,1))

    # make matrix with all possible single-defect perturbations
    states_defect_all = (np.tile(states, (N,1)) + np.diag(np.ones(N, dtype=int))) % 2
    # calculate perturbed next-timestep state vector (N times in parallel)
    states_defect_all_next = np.empty_like(states_defect_all)
    for tt, states_defect in enumerate(states_defect_all):
        states_defect_next = cpl.evolve(states_defect[np.newaxis,:], timesteps=2, apply_rule=lambda n, c, t: cpl.nks_rule(n, rule))[-1]
        states_defect_all_next[tt] = states_defect_next

    # compile Jacobian with XOR operator (the Boolean derivative)
    J = (states_all_next + states_defect_all_next) % 2

    # return transpose (to match definition)
    if return_next:
        return J.T, states_next
    return J.T

def defect_diameter(
    graph:ig.Graph,
    model:LLNA,
    states:NDArray[np.int_],
    T:int,
    norm:bool=True
) -> Union[NDArray[np.int_] ,NDArray[np.floating]]:
    """
    Calculates the configuration-space interpretation of the Lyapunov exponent after T time steps.
    This is defined by the diameter of the subgraph of all affected nodes. If normalised, it is divided by the diameter of the original graph.
    The value is given for the perturbance of each of the nodes (in the order of the nodes in the graph object)

    Parameters
    ----------
    graph : igraph.Graph
        Network (graph) used as the topology for the automaton. Will be interpreted as a undirected graph.
    model : src.automata.LLNA
        Life-like network automaton (custom class). Envelops the rules that govern the automaton.
    states : numpy.ndarray
        Initial configuration of the LLNA.
    T : int
        The number of time steps over which the defect diameter is calculated and normalised
    norm : bool
        Defaults to True. If False, the defect diameters are not normalised (divided by the diameter of the parent graph).

    Returns
    -------
    diameters : numpy.ndarray
        An array of length T with the defect diameters related to a defect in each of the N nodes.
        If norm==False, the diameters are not normalised (divided by the diameter of the parent graph).
    
    NOTE: this function turns out not to be terribly useful, actually.
    """
    N = len(states)

    # get ID of edges (bidirectional)
    graph.to_directed()
    edges = tc.tensor(graph.get_edgelist()).T
    graph.to_undirected()

    # parallellise all possible single defects
    states_original = np.tile(states, (N,1))
    states_perturbed = (states_original + np.diag(np.ones(N, dtype=int))) % 2
    # dimensions of states: [defect_index, time, node]
    states_original = np.array(model.forward(edges, tc.tensor(states_original), T=T), dtype=int)
    states_perturbed = np.array(model.forward(edges, tc.tensor(states_perturbed), T=T), dtype=int)

    # find defects and indicate nodes with a defect history with a 1
    deltas = (states_original + states_perturbed) % 2
    deltas_cum = np.cumsum(deltas, axis=1)
    deltas_cum = np.clip(deltas_cum, 0, 1)

    # define subgraph for each of the defects at time step T
    diameters = []
    for delta_cum in deltas_cum[:,-1,:]:
        delta_nodes, = np.where(delta_cum)
        subG = graph.subgraph(delta_nodes)
        diameter = subG.diameter()
        diameters += [diameter]
    diameters = np.array(diameters)
    if norm:
        diameters = diameters/graph.diameter()
    return diameters

def hamming_weight(
    resolution:int,
    B_set:Union[NDArray[np.int_], Sequence[int]],
    S_set:Union[NDArray[np.int_], Sequence[int]],
    degree:int,
    norm:bool=True,
    iso=True
) -> Union[float, int]:
    """
    Calculate the (normalised) Hamming weight of the local update rule for a particular node degree. There is no need for a LLNA object (so the calculation is generally faster).
    NOTE: this defaults to iso=True right now.

    Parameters
    ----------
    resolution : int
        Positive integer indicating the resolution of the local update rule
    B_set : list
        list of integers corresponding to the indices of the activated density intervals in the B set
    S_set : list
        list of integers corresponding to the indices of the activated density intervals in the S set
    degree : int
        The degree of the node you want to calculate the Hamming weight for
    norm : bool
        True by default. Maps the Hamming weight to a value from 0 to 1.
    iso : bool
        True by default. Set to False if non-isomorphic density intervals are used in the LLNA definition.

    Returns
    -------
    HW : float or int
        The (normalised) Hamming weight
    """
    # NOTE: so far this is limited to degree 1022
    # this is identical to the Langton parameter!
    if (degree < 1) or (degree > 1022):
        raise ValueError(f"Degree {degree} not allowed. The degree must be a non-zero natural number smaller than 1023.")
    if iso and resolution % 2 == 0:
        raise ValueError(f"Resolution {resolution} is not possible when iso=True. Choose an odd positive integer.")
    if (B_set and (np.max(B_set) >= resolution)) or (S_set and (np.max(S_set) >= resolution)):
        raise ValueError(f"Resolution {resolution} is too small for the provided update intervals.")
    # all possible densities for this degree
    rhos = np.linspace(0, 1, degree+1)
    # the number of configurations that correspond to this density
    configs_per_rho = np.array([comb(degree, k, exact=False) for k in range(degree + 1)])
    # make sure this sums to 2**(degree)
    expected_sum = 2**degree
    configs_per_rho *= expected_sum / np.sum(configs_per_rho)
    # make truthtable for resp. dead and living central nodes
    rho_intervals = _interval_encoding(resolution, rhos[np.newaxis,:], iso=iso)[0].argmax(axis=1)
    born_truthtable = np.array([(rho in B_set) for rho in rho_intervals])
    survive_truthtable = np.array([(rho in S_set) for rho in rho_intervals])
    # weighted sum of all truthtable outputs
    born_tt_sum = np.sum(configs_per_rho * born_truthtable)
    survive_tt_sum = np.sum(configs_per_rho * survive_truthtable)
    # born and survive set are both make up the total truth table
    HW = born_tt_sum + survive_tt_sum
    if norm:
        tt_size = 2**(degree+1)
        HW_norm = HW / tt_size
        return HW_norm
    return int(HW)

def mean_field_dens_propagation(
    resolution: int,
    B_set:Union[NDArray[np.int_], Sequence[int]],
    S_set:Union[NDArray[np.int_], Sequence[int]],
    current_dens:Union[float, NDArray],
    degree:int,
    iso:bool=True
) -> NDArray:
    """
    Calculates the average density of a randomly chosen neighbourhood (with a particular degree) at time step 1,
    after evolving according to the rule defined by the resolution, B set and S set,
    if you know the average density over the entire network at the initial configuration (time step 0).
    Note that we suppose that the initial configuration is chosen randomly.

    Parameters
    ----------
    resolution : int
        Positive integer indicating the resolution of the local update rule
    B_set : list
        list of integers corresponding to the indices of the activated density intervals in the B set
    S_set : list
        list of integers corresponding to the indices of the activated density intervals in the S set
    current_dens : float
        The network-wide average density at the initial configuration.
    degree : int
        The degree of the node you want to calculate the expected neighbourhood density for
    iso : bool
        True by default. Set to False if non-isomorphic density intervals are used in the LLNA definition.

    Returns
    -------
    next_dens : float
        The expected value of the neighbourhood density in time step 1
    """
    # check that the input values make sense
    if (degree < 1) or (degree > 1022):
        raise ValueError(f"Degree {degree} not allowed. The degree must be a non-zero natural number smaller than 1023.")
    if iso and resolution % 2 == 0:
        raise ValueError(f"Resolution {resolution} is not possible when iso=True. Choose an odd positive integer.")
    if (B_set and (np.max(B_set) >= resolution)) or (S_set and (np.max(S_set) >= resolution)):
        raise ValueError(f"Resolution {resolution} is to small for the provided update intervals.")
    current_dens = np.atleast_1d(current_dens)
    if (np.min(current_dens) < 0) or (np.max(current_dens) > 1):
        raise ValueError(f"Average state density must be between 0 and 1. Now the maximum is {np.max(current_dens)} and the minimum is {np.max(current_dens)}.")

    # find all the possible rho_i values for this degree
    rhos = np.linspace(0, 1, degree + 1)
    # check in which interval they are situated
    rho_intervals = _interval_encoding(resolution, rhos[np.newaxis, :], iso=iso)[0].argmax(axis=1)
    # find what this outputs to
    born_truthtable = np.array([int(rho in B_set) for rho in rho_intervals])
    survive_truthtable = np.array([int(rho in S_set) for rho in rho_intervals])
    # calculate the binomial probabilities for various sums q, based on the current average density
    binom_q = binom.pmf(np.arange(degree + 1), degree, current_dens[:, np.newaxis])
    # calculate the weighted born and survive truthtables (based on probability of finding the central node alive)
    born_truthtable_weighted = born_truthtable * (1 - current_dens)[:, np.newaxis]
    survive_truthtable_weighted = survive_truthtable * current_dens[:, np.newaxis]
    # sum over all possible sum values
    next_dens = np.sum(binom_q * (born_truthtable_weighted + survive_truthtable_weighted), axis=1)
    # return the next density and make sure it's between 0 and 1
    return np.clip(next_dens, 0, 1)

def mean_field_slope(resolution, B_set, S_set, degree, rho_star=0.5, iso=True):
    # TODO this currently does not work well for rho_star 0 or rho_star 1 (division by zero)
    # TODO: add documentation
    # TODO: add numpy magic (right now it's slow)
    if (degree < 1) or (degree > 1022):
        raise ValueError(f"Degree {degree} not allowed. The degree must be a non-zero natural number smaller than 1023.")
    if iso and resolution % 2 == 0:
        raise ValueError(f"Resolution {resolution} is not possible when iso=True. Choose an odd positive integer.")
    if (B_set and (np.max(B_set) >= resolution)) or (S_set and (np.max(S_set) >= resolution)):
        raise ValueError(f"Resolution {resolution} is to small for the provided update intervals.")
    if (rho_star < 0) or (rho_star > 1):
        raise ValueError(f"Equilibrium state density `rho_star` must be between 0 and (not {rho_star}).")

    # find all the possible rho_i values for this degree
    rhos = np.linspace(0, 1, degree + 1)
    # check in which interval they are situated
    rho_intervals = _interval_encoding(resolution, rhos[np.newaxis, :], iso=iso)[0].argmax(axis=1)
    # find binomium elements
    binomium_factor = np.array([comb(degree, q) for q in range(0, degree+1)])
    # find what this outputs to
    born_truthtable = np.array([int(rho in B_set) for rho in rho_intervals])
    survive_truthtable = np.array([int(rho in S_set) for rho in rho_intervals])
    # calculate the binomial probabilities for various sums q, based on the current average density
    sum_prefactor_born = np.array([rho_star**(q - 1) * (1 - rho_star)**(degree - q) * (q - rho_star * (1 + degree)) for q in range(degree + 1)])
    sum_prefactor_survive = np.array([rho_star**q * (1 - rho_star)**(degree - q - 1) * (1 + q - rho_star * (1 + degree)) for q in range(degree + 1)])
    # calculate the weighted born and survive truthtables (based on probability of finding the central node alive)
    born_truthtable_weighted = born_truthtable * sum_prefactor_born
    survive_truthtable_weighted = survive_truthtable * sum_prefactor_survive
    # sum over all possible sum values
    slope = np.sum(binomium_factor * (born_truthtable_weighted + survive_truthtable_weighted))
    # return the next density and make sure it's between 0 and 1
    return slope

def derrida_map_analytical(
    resolution:int,
    B_set:Union[NDArray[np.int_], Sequence[int]],
    S_set:Union[NDArray[np.int_], Sequence[int]],
    delta0:Union[float, NDArray],
    degree:int,
    init_config_dens:float=0.5,
    iso:bool=True
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
        raise ValueError(f"Degree {degree} not allowed. The degree must be a non-zero natural number smaller than 1023.")
    if iso and resolution % 2 == 0:
        raise ValueError(f"Resolution {resolution} is not possible when iso=True. Choose an odd positive integer.")
    if (B_set and (np.max(B_set) >= resolution)) or (S_set and (np.max(S_set) >= resolution)):
        raise ValueError(f"Resolution {resolution} is to small for the provided update intervals.")
    delta0 = np.atleast_1d(delta0)
    if (np.min(delta0) < 0) or (np.max(delta0) > 1):
        raise ValueError(f"Normalised Hamming weight delta0 must be between 0 and 1. Now the maximum is {np.max(delta0)} and the minimum is {np.max(delta0)}.")
    if (init_config_dens < 0) or (init_config_dens > 1):
        raise ValueError(f"Density rho must be between 0 and 1, not {init_config_dens}.")
    
    # define the big XOR function that is at the end of all the summations
    def function_xor(
        resolution,         # r
        B_set,              # B
        S_set,              # S
        centre_state,       # s
        num_living,         # q
        centre_toggle,      # c
        num_killer_toggles, # t
        num_defects,        # d
        degree   ,          # k
        iso=True            # isomorphic interval encoding
    ):
        # but B_set and S_set in a list
        B_and_S_set = [B_set, S_set]
        # calculate the density interval for the default and the defect case
        rho = num_living / degree
        rho_defect = (num_living - 2*num_killer_toggles + num_defects) / degree
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
        return rho0**centre_state * (1-rho0)**(1-centre_state)
    def prob_q_living_neighbours(num_living, degree, rho0):
        return binom.pmf(num_living, degree, rho0)
    def prob_centre_node_toggle(centre_toggle, delta0):
        return delta0**centre_toggle * (1-delta0)**(1-centre_toggle)
    def prob_d_toggles(num_defects, degree, delta0):
        return binom.pmf(num_defects, degree, delta0)
    def prob_killer_toggle(num_killer_toggles, degree, num_living, num_defects):
        return hypergeom.pmf(num_killer_toggles, degree, num_living, num_defects)
    
    # sum over all the possible combinations
    # TODO: this is coded pretty badly with all these loops ...
    delta1 = np.zeros_like(delta0)
    for centre_state in [0,1]:
        prob1 = prob_centre_node_alive(centre_state, init_config_dens)
        for num_living in range(degree+1):
            prob2 = prob_q_living_neighbours(num_living, degree, init_config_dens)
            for centre_toggle in [0,1]:
                prob3 = prob_centre_node_toggle(centre_toggle, delta0)
                for num_defects in range(degree+1):
                    prob4 = prob_d_toggles(num_defects, degree, delta0)
                    min_t = max(0, num_defects + num_living - degree)
                    max_t = min(num_defects, num_living)
                    for num_killer_toggles in range(min_t, max_t+1):
                        prob5 = prob_killer_toggle(num_killer_toggles, degree, num_living, num_defects)
                        function_output = function_xor(resolution, B_set, S_set, centre_state, num_living, centre_toggle, num_killer_toggles, num_defects, degree)
                        value = prob1 * prob2 * prob3 * prob4 * prob5 * function_output
                        delta1 += value
    return np.clip(delta1, 0, 1)

def nbh_sensitivity_naive(
    resolution: int,
    B_set:Union[NDArray[np.int_], Sequence[int]],
    S_set:Union[NDArray[np.int_], Sequence[int]]
) -> float:
    """
    Returns a value between 0 and 1, indicating how sensitive the output of this LLNA is to slightly changing the neighbourhood density. This function does not require an LLNA object, which makes the calculation much faster.
    NOTE: this is a naive definition that does not take into account the degree distribution and the state density distribution.

    Parameters
    ----------
    resolution : int
        Positive integer indicating the resolution of the local update rule
    B_set : list or numpy.ndarray
        list of integers corresponding to the indices of the activated density intervals in the B set
    S_set : list or numpy.ndarray
        list of integers corresponding to the indices of the activated density intervals in the S set

    Returns
    -------
    nbh_sens_naive : float
        Value ranging from 0 to 1, indicating low resp. high sensitivity.
    """
    B_set = np.asarray(B_set, dtype=int)
    S_set = np.asarray(S_set, dtype=int)
    # turn interval index into bits
    B_set_bin = np.zeros(resolution, dtype=int)
    S_set_bin = np.zeros(resolution, dtype=int)
    if B_set.size: B_set_bin[B_set] = 1
    if S_set.size: S_set_bin[S_set] = 1
    # perform XOR with shifted arrays
    borders_B = np.logical_xor(B_set_bin[1:], B_set_bin[:-1]).sum()
    borders_S = np.logical_xor(S_set_bin[1:], S_set_bin[:-1]).sum()
    # return total number of edges
    nbh_sens_naive = (borders_B + borders_S)/(resolution-1)/2
    return nbh_sens_naive

def id_sensitivity_naive(
    resolution: int,
    B_set:Union[NDArray[np.int_], Sequence[int]],
    S_set:Union[NDArray[np.int_], Sequence[int]]
) -> float:
    """
    Returns a value between 0 and 1, indicating how sensitive the output of this LLNA is to changing the value of a particular node. This function does not require an LLNA object, which makes the calculation much faster.
    NOTE: this is a naive definition that does not take into account the degree distribution and the state density distribution.

    Parameters
    ----------
    resolution : int
        Positive integer indicating the resolution of the local update rule
    B_set : list
        list of integers corresponding to the indices of the activated density intervals in the B set
    S_set : list
        list of integers corresponding to the indices of the activated density intervals in the S set

    Returns
    -------
    id_sens_sens : float
        Value ranging from 0 to 1, indicating low resp. high sensitivity.
    """
    B_set = np.array(B_set)
    S_set = np.array(S_set)
    # turn interval index into bits
    B_set_bin = np.zeros(resolution, dtype=int)
    S_set_bin = np.zeros(resolution, dtype=int)
    if B_set.size: B_set_bin[B_set] = 1
    if S_set.size: S_set_bin[S_set] = 1
    borders = np.logical_xor(B_set_bin, S_set_bin).sum()
    id_sens = borders / resolution
    return id_sens

def boolean_sens(
    resolution:int,
    B_set:Union[NDArray[np.int_], Sequence[int]],
    S_set:Union[NDArray[np.int_], Sequence[int]],
    degree:int,
    norm_degree:bool=False,
    current_dens:float=0.5,
    iso:bool=True
) -> float:
    """
    Calculate the (normalised) Boolean sensitivity of the local update rule for a particular node degree. There is no need for a LLNA object (so the calculation is generally faster).
    NOTE: this defaults to iso=True right now.
    TODO: this is programmed really sloppily (cause it's mostly copied from previous methods), but it works.

    Parameters
    ----------
    resolution : int
        Positive integer indicating the resolution of the local update rule
    B_set : list or numpy.ndarray
        list of integers corresponding to the indices of the activated density intervals in the B set
    S_set : list or numpy.ndarray
        list of integers corresponding to the indices of the activated density intervals in the S set
    degree : int
        The degree of the node you want to calculate the Boolean sensitivity for
    norm_degree : bool
        False by default. Maps the Boolean sensitivity to a value from 0 to 1. Note that this is typically not desired (because then it no longer corresponds to the Derrida coefficient)
    iso : bool
        True by default. Set to False if non-isomorphic density intervals are used in the LLNA definition.

    Returns
    -------
    BS : float
        The (normalised) Boolean sensitivity
    """
    if (degree < 1) or (degree > 1022):
        raise ValueError(f"Degree {degree} not allowed. The degree must be a non-zero natural number smaller than 1023.")
    if iso and resolution % 2 == 0:
        raise ValueError(f"Resolution {resolution} is not possible when iso=True. Choose an odd positive integer.")
    if (B_set and (np.max(B_set) >= resolution)) or (S_set and (np.max(S_set) >= resolution)):
        raise ValueError(f"Resolution {resolution} is to small for the provided update intervals.")
    # identity sensitivity
    def _id_sens(resolution, B_set, S_set, degree, iso=True):
        """
        returns an array with length degree+1, containing 0s or 1s
            0: the corresponding state density has identical outputs when flipping the central node
            1: the corresponding state density has different outputs when flipping the central node
        """
        # find density and which interval it belongs to (from one-hot vector)
        rhos = np.linspace(0, 1, degree+1)
        # make truthtable for resp. dead and living central nodes
        rho_intervals = _interval_encoding(resolution, rhos[np.newaxis,:], iso=iso)[0].argmax(axis=1)
        B_truthtable = np.array([int(rho in B_set) for rho in rho_intervals])
        S_truthtable = np.array([int(rho in S_set) for rho in rho_intervals])
        IS = (B_truthtable ^ S_truthtable)
        return IS

    # neighbourhood sensitivity for increasing densitity
    def _nbh_sens_inc(resolution, B_set, S_set, si, degree, iso=True):
        # find all possible densities except density 1 (which cannot increase)
        rhos = np.linspace(0, 1, degree+1)[:-1]
        # find which interval these densities belong to
        rho_intervals = _interval_encoding(resolution, rhos[np.newaxis,:], iso=iso)[0].argmax(axis=1)
        # find INCREASED density and which interval it belongs to (from one-hot vector)
        rhos_inc = np.linspace(0, 1, degree+1)[1:]
        rho_inc_intervals = _interval_encoding(resolution, rhos_inc[np.newaxis,:], iso=iso)[0].argmax(axis=1)
        # check whether they have the same response
        B_or_S_set = [B_set, S_set][int(si)]
        B_or_S_truthtable = np.array([int(rho in B_or_S_set) for rho in rho_intervals])
        B_or_S_truthtable_inc = np.array([int(rho_inc in B_or_S_set) for rho_inc in rho_inc_intervals])
        NS_inc = (B_or_S_truthtable ^ B_or_S_truthtable_inc)
        # add a zero at the end and return
        return np.append(NS_inc, 0)

    # neighbourhood sensitivity for decreasing densitity
    def _nbh_sens_dec(resolution, B_set, S_set, si, degree, iso=True):
        # find all possible densities except density 0 (which cannot decrease)
        rhos = np.linspace(0, 1, degree+1)[1:]
        # find which interval these densities belong to
        rho_intervals = _interval_encoding(resolution, rhos[np.newaxis,:], iso=iso)[0].argmax(axis=1)
        # find DECREASED density and which interval it belongs to (from one-hot vector)
        rhos_dec = np.linspace(0, 1, degree+1)[:-1]
        rho_dec_intervals = _interval_encoding(resolution, rhos_dec[np.newaxis,:], iso=iso)[0].argmax(axis=1)
        # check whether they have the same response
        B_or_S_set = [B_set, S_set][int(si)]
        B_or_S_truthtable = np.array([int(rho in B_or_S_set) for rho in rho_intervals])
        B_or_S_truthtable_dec = np.array([int(rho_dec in B_or_S_set) for rho_dec in rho_dec_intervals])
        NS_dec = (B_or_S_truthtable ^ B_or_S_truthtable_dec)
        # add a zero at the start and return
        return np.append(0, NS_dec)
    
    # overall neighbourhood sensitivity
    def _nbh_sens(resolution, B_set, S_set, si, degree, iso=True):
        # all possible q values
        sum_values = np.arange(0, degree+1)
        NS_dec = _nbh_sens_dec(resolution, B_set, S_set, si, degree, iso=iso)
        NS_inc = _nbh_sens_inc(resolution, B_set, S_set, si, degree, iso=iso)
        return sum_values*NS_dec + (degree-sum_values)*NS_inc
    
    # combinatorial elements
    binom_q = binom.pmf(np.arange(degree + 1), degree, current_dens)
    binom_s = [1-current_dens, current_dens]

    # use functions above to find BS
    prefactor = 1
    if norm_degree:
        prefactor /= (degree+1)
    summation = 0
    for si in [0,1]:
        IS = _id_sens(resolution, B_set, S_set, degree, iso=iso)
        NS = _nbh_sens(resolution, B_set, S_set, si, degree, iso=iso)
        term = np.sum((IS + NS) * binom_q * binom_s[si])
        summation += term
    BS = prefactor * summation
    return BS

def average_metric_over_degrees(
    resolution:int,
    B_set:Union[NDArray[np.int_], Sequence[int]],
    S_set:Union[NDArray[np.int_], Sequence[int]],
    metric_function,
    graph:ig.Graph
) -> float:
    """
    A function that takes the weighted sum of a particular metric (HW or BS or ...) over all possible degrees in the graph.

    Parameters
    ----------
    resolution : int
        Positive integer indicating the resolution of the local update rule
    B_set : list
        list of integers corresponding to the indices of the activated density intervals in the B set
    S_set : list
        list of integers corresponding to the indices of the activated density intervals in the S set
    metric_function : function
        The function that calculates the metric of interest (e.g. HW or BS). Should take exactly four arguments: resolution, B_set, S_set, degree.
    graph : igraph.Graph
        Network (graph) used as the topology for the automaton. Will be interpreted as a undirected graph.

    Returns
    -------
    id_sens_sens : float
        Value ranging from 0 to 1, indicating low resp. high sensitivity.
    """
    # calculate degree distribution
    degrees = graph.degree()
    bins = range(1,max(degrees)+2)
    degree_ns, _ = np.histogram(degrees, bins=bins)
    degrees = list(bins[:-1])
    # calculate weighted average
    metric_per_degree = np.array([degree_n*metric_function(resolution, B_set, S_set, int(degree)) for degree_n, degree in zip(degree_ns, degrees)])
    average = np.sum(metric_per_degree) / graph.vcount()
    return average

def calculate_Yt(
    graph:ig.Graph,
    model:LLNA,
    states:NDArray[np.int_], T:int
) -> NDArray[np.floating]:
    """
    Calculates the perturbation of the unit defect sphere Y0 in tangent space starting from the point in configuration space indicated by the states array, after T time steps. This is used to calculate the tangent-space interpretation of the Lyapunov exponent of a cellular automaton or network automaton.

    Parameters
    ----------
    graph : igraph.Graph
        Network (graph) with N nodes used as the topology for the automaton. Will be interpreted as a undirected graph.
    model : src.automata.LLNA
        Life-like network automaton (custom class). Envelops the rules that govern the automaton.
    states : numpy.ndarray
        Initial configuration (length N) of the LLNA.
    T : int
        The number of time steps over which the defect diameter is calculated and normalised

    Returns
    -------
    Yt : numpy.ndarray
        N x N matrix were each column represents the perturbation of an initial unit defect in tangent space.
    """
    N = len(states)
    # start with unit sphere (with 64 bit precision to avoid numerical errors)
    Yt = np.diag(np.ones(N, dtype=np.float64))
    # multiply subsequent Jacobians
    for _ in range(T):
        J, states = jacobian(graph, model, states, return_next=True)
        Yt = np.matmul(J.astype(np.float64), Yt) # no mod 2! And no integers to avoid overflow!
    return Yt

def lyapunov_spectrum(
    Yt:Union[NDArray[np.floating], NDArray[np.int_]],
    T:int
) -> NDArray[np.floating]:
    """
    Calculates the Lyapunov spectrum in the tangent-space interpretation, by taking the natural logarithm of the singular values of the evolved unit-perturbation sphere.

    Parameters
    ----------
    Yt : numpy.ndarray
        N x N matrix were each column represents the perturbation of an initial unit defect in tangent space.
    T : int
        The number of time steps over which the defect diameter is calculated and normalised

    Returns
    -------
    lambdas : numpy.ndarray
        Array of length N, containing the Lyapunov coefficient for an initial minimal defect in each of the N dimensions of the discrete dynamical system.
    """
    # Gamma = np.matmul(Yt, Yt.T)
    # # Symmetrize the matrix to ensure it's exactly symmetric (avoid numerical errors)
    # Gamma = (Gamma + Gamma.T) / 2
    # # use the eigh method (which is optimised for symmetric matrices). Note that we only want positive eigenvalues.
    # # TODO: this function returns the eigenvalues in ascending order, i.e. it loses information on which node is affected!
    # # TODO: this function probably encounters numerical issues!
    # Lambdas_squared = np.abs(np.linalg.eigvalsh(Gamma))
    # # Eq. 13 in Vispoel et al (2024)
    # lambdas = np.log(Lambdas_squared)/(2*T)
    singulars = np.linalg.svd(Yt, compute_uv=False)
    lambdas = np.log(singulars)/T
    return lambdas

def lyapunov_spectrum_analytical(
    rule:int,
    N:int,
    return_finite_pct=False
) -> Union[NDArray[np.floating], Tuple[NDArray[np.floating], float]]:
    """
    Calculates the Lyapunov spectrum in the tangent-space interpretation, by taking the natural logarithm of the singular values of the evolved unit-perturbation sphere. Here we make use of the (supposed) fact that we are calculating the singular values for an ECA with a constant Jacobian. This Jacobian is circulant, which in turn allows for an analytical expression. In so doing, we avoid any numerical instabilities that tend to arise in the numerical approach.
    """
    # exceptions
    if not eca_has_constantJ(rule):
        raise ValueError(f"Rule {rule} does not have a constant Jacobian.")
    if N<0:
        raise ValueError(f"The number of nodes must be a natural number. Got N={N}.")
    # find constants
    J = jacobian_ECA(rule, np.zeros(N), return_next=False)
    triplet = J[1][0:3]
    c0 = triplet[1]
    c1 = triplet[0]
    cN_1 = triplet[2]
    cc0 = c0**2 + c1**2 + cN_1**2
    cc1 = c0*c1 + cN_1*c0
    cc2 = cN_1*c1
    # calculate singular values

    k_values = np.arange(N)
    eigenvalues = cc0 + 2 * cc1 * np.cos(2 * k_values * np.pi / N) + 2 * cc2 * np.cos(4 * k_values * np.pi / N)
    if np.any(eigenvalues) < 0:
        raise ValueError("Due to a numerical error, there are negative values among the eigenvalues.")
    singular_values = np.sqrt(eigenvalues)
    # eliminate singular values of zero (and negative ones from numerical problems)
    nonzero_singular_values = singular_values[singular_values>0]
    lyapunov_values = np.log(nonzero_singular_values)
    if return_finite_pct:
        finite_pct = round(len(nonzero_singular_values) / N * 100)
        return lyapunov_values, finite_pct
    return lyapunov_values

def get_derrida_arrays(
    graph:ig.Graph,
    model:LLNA,
    points_per_rho:int=1,
    num_init_configs:Optional[int]=None,
    init_configs:Optional[NDArray]=None,
    return_until_dens:float=1.0
) -> Tuple[NDArray, NDArray]:
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
            raise ValueError(f"When manually entering the initial configurations, the kwarg `num_init_config` must be None.")
        if init_configs.shape[-1] != N:
            raise ValueError(f"The provided initial configuration(s) should have length {N}.")
        if init_configs.ndim > 2:
            raise ValueError(f"The provided initial configuration(s) should have either 1 or 2 dimensions.")
        if init_configs.ndim < 2: # fix dimensions if just a single array is given
            init_configs = init_configs[np.newaxis, :]
        num_init_configs = init_configs.shape[0]
    else: # make a single random initial configuration
        if num_init_configs is None:
            num_init_configs = 1
        init_configs= np.random.randint(0,2,size=(num_init_configs,N))

    # get a long array of unique defects
    if (points_per_rho*num_init_configs) > N:
        raise Exception(f"The number of data points per normalised hamming weight cannot be larger than or equal to {N}, because there are not that many unique combinations of defect arrays.\nLower the points per density and/or the number of initial configuration, and/or increase the number of nodes in the network.")
    # calculate the cutoff value of the defect density
    if not (1/N <= return_until_dens <= 1.):
        raise ValueError(f"The kwarg return_until_dens must be a value between {1/N} and 1.")
    max_ones_per_array = int(N*return_until_dens)
    # take case where max_ones is N, which is not helpful
    max_ones_per_array = min(N-1, max_ones_per_array)
    # create array of max ones
    max_ones_per_array_range = range(1,max_ones_per_array+1)
    # create defect array using a helper function
    defects_all = np.vstack([_random_ones_arrays(N, points_per_rho*num_init_configs, ones_per_array) for ones_per_array in max_ones_per_array_range])

    # copy the initial configurations points_per_rho times
    num_rhos = len(max_ones_per_array_range)
    init_configs_all = np.tile(init_configs, (points_per_rho*num_rhos,1))
    # add the defects to the initial configurations in ascending order of number of defects
    init_configs_defect_all = (init_configs_all + defects_all) % 2

    # get ID of edges (bidirectional)
    graph.to_directed()
    edges = tc.tensor(graph.get_edgelist()).T
    graph.to_undirected()

    # run the model for a single time step for the various initial conditions
    next_configs = np.array(model.step(edges, tc.tensor(init_configs)), dtype=int)
    # copy the output as many times as required (for points_per_rho)
    next_config_all = np.tile(next_configs, (points_per_rho*num_rhos,1))
    # run the model for a single time step for all the defected initial conditions
    next_config_defect_all = np.array(model.step(edges, tc.tensor(init_configs_defect_all)), dtype=int)

    # find the normalised Hamming distance between both new configurations
    input_defect_density  = np.mean(defects_all, axis=1)
    output_defect_density = np.mean((next_config_all + next_config_defect_all) % 2, axis=1)

    return input_defect_density, output_defect_density

def calculate_derrida_coefficient(rho_t_array: NDArray, rho_tplus1_array: NDArray, cutoff_rho_t: float = 0.05) -> np.floating:
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
        raise ValueError(f"Maximum value of rho_t_array is less than the requested kwarg value `cutoff_rho_t`={cutoff_rho_t}.")
    rho_t_array = rho_t_array[rho_t_array <= cutoff_rho_t]
    rho_tplus1_array = rho_tplus1_array[:len(rho_t_array)]
    # calculate slope using the least squares method from this subset of densities
    derrida_coefficient = np.sum(rho_t_array * rho_tplus1_array) / np.sum(rho_t_array * rho_t_array)

    # return Derrida coefficient. NOTE that different sources use different definitions
    return derrida_coefficient

def derrida_spline_and_roots(inputs: np.ndarray, outputs: np.ndarray, num_bins: int = 20) -> Tuple[CubicSpline, List[float]]:
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
    x_binned = [x_sorted[digitized == i].mean() for i in range(1, num_bins) if len(x_sorted[digitized == i]) > 0]
    y_binned = [np.median(y_sorted[digitized == i]) for i in range(1, num_bins) if len(y_sorted[digitized == i]) > 0]

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
                root = root_scalar(diagonal_crossing, bracket=[x_fine[i], x_fine[i + 1]], method='brentq').root
                roots.append(root)
            except ValueError:
                pass  # Skip if no valid root is found
    return spline, roots

def eca_has_constantJ(eca:int) -> bool:
    # TODO: this is hard-coded for now.
    if not _is_eca(eca):
        raise Exception(f"ECA '{eca}' is not recognised as an elementary cellular automaton. Choose an integer from 0 to 255.")
    all_constantJ_rules = [0, 255, 85, 170, 51, 204, 102, 153, 15, 240, 90, 165, 60, 195, 105, 150]
    if eca not in all_constantJ_rules:
        return False
    return True

def median_and_percentiles_over_ensemble(arrays, delta_t, lower_percentile=0.25, upper_percentile=0.75):
    """
    Simple function to find the convergence value of an ensemble of time series
    """
    if arrays.ndim != 2:
        raise ValueError(f"The input array has {arrays.ndim} dimensions instead of 2.")
    if (lower_percentile > 0.5) or (upper_percentile < 0.5):
        raise ValueError("The median is not within the percentiles.")
    final_timesteps = arrays[-delta_t:]
    # take median over all values (no distinction between time and ensemble dimension)
    median = np.median(final_timesteps)
    lw_perc = np.quantile(final_timesteps, lower_percentile)
    up_perc = np.quantile(final_timesteps, upper_percentile)
    return median, lw_perc, up_perc

def switch_life_to_higher_node_property_values(node_property_values,
                                               init_config,
                                               number_of_switches,
                                               kill_low_values=True,
                                               random_seed=None,
                                               return_sorted_values=False):
    """
    Modifies an initial configuration by switching the states of nodes based on their property values.
    By default, nodes with low property values that are alive are set to dead, while nodes with high 
    property values that are dead are set to alive. The total number of alive nodes remains constant.

    Parameters
    ----------
    node_property_values : numpy.ndarray
        Array containing the property value for each node (e.g., degree, centrality). Must have the 
        same length as init_config.
    init_config : numpy.ndarray
        Initial configuration array with binary states (0 or 1) for each node.
    number_of_switches : int
        Number of nodes to switch from alive to dead (low property values) and from dead to alive 
        (high property values). Must be a non-negative integer not exceeding half the number of nodes.
    kill_low_values : bool
        Defaults to True. If True, nodes with low property values are set to dead (0) and nodes with 
        high property values are set to alive (1). If False, the behavior is reversed.
    random_seed : int, optional
        Random seed for reproducibility of the shuffling process. If None (default), the random 
        state is not modified.

    Returns
    -------
    init_config_switched : numpy.ndarray
        Modified configuration array with switched node states based on property values.

    Notes
    -----
    When multiple nodes share the same property value, their order is randomized within that group 
    to avoid introducing spurious correlations.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    if number_of_switches<0:
        raise ValueError(f"The number of switches must be a nutural number. Not {number_of_switches}.")
    # get all values of this metric
    num_nodes = len(init_config)
    if len(node_property_values) != num_nodes:
        raise ValueError(f"The number of nodes and the number of initial states do not match.")
    if number_of_switches > num_nodes//2:
        raise ValueError(f"You cannot switch places of more than half of the nodes.")
    default_value = 0
    if not kill_low_values:
        default_value = 1

    # get the indices of all nodes sorted by the node property
    sorted_node_indices = np.argsort(node_property_values)
    sorted_values = node_property_values[sorted_node_indices]
    # find unique values
    unique_values = np.unique(node_property_values)
    # if there not all values are unique, randomize the order within each subgroup with the same node property value
    if len(unique_values) != num_nodes:
        # Randomize within groups (to avoid other correlations)
        shuffled_sorted_node_indices = np.empty_like(sorted_node_indices)
        start = 0
        for unique_value in unique_values:
            # find where the indices correspond to the current node property value
            val_indices = np.where(sorted_values == unique_value)[0]
            # get the subarray of values coresponding to this value
            group = sorted_node_indices[val_indices]
            # randomise the group
            np.random.shuffle(group)
            # glue the pieces back together
            shuffled_sorted_node_indices[start:start + len(group)] = group
            start += len(group)
        # order the initial configuration according to the shuffled value-sorted indices
        init_config_shuffle_ordered = init_config[shuffled_sorted_node_indices]
    else:
        shuffled_sorted_node_indices = sorted_node_indices
        init_config_shuffle_ordered = init_config[shuffled_sorted_node_indices]
        
    # find the shuffled indices with low node property whose node is alive and kill 'em dead (by default)
    indices_where_alive = np.where(init_config_shuffle_ordered==1-default_value)[0][:number_of_switches]
    init_config_shuffle_ordered[indices_where_alive] = default_value-0
    # same for shuffled indices with high node property of dead nodes that are brought to life (by default)
    indices_where_dead  = np.where(init_config_shuffle_ordered==default_value-0)[0]
    number_of_dead = len(indices_where_dead)
    indices_where_dead = indices_where_dead[(number_of_dead-number_of_switches):]
    init_config_shuffle_ordered[indices_where_dead] = 1-default_value
    # undo the ordering
    init_config_switched = init_config_shuffle_ordered[np.argsort(shuffled_sorted_node_indices)]

    if return_sorted_values:
        sorted_values = node_property_values[shuffled_sorted_node_indices]
        return init_config_switched, sorted_values
    return init_config_switched

def init_config_with_dens(N, dens):
    """
    Initialize a configuration array with a given density of ones.

    Parameters:
    N (int): The size of the array.
    dens (float): The density of ones in the array (between 0 and 1).

    Returns:
    numpy.ndarray: A shuffled array of size N with the specified density of ones.
    """
    s0 = np.zeros(N, dtype=int)
    s0[:np.round(dens*N).astype(int)] = 1.
    np.random.shuffle(s0)
    return s0

#%% helper functions

def _interval_encoding(resolution:int, rhos, iso=True):
    if not iso: # classic psuedo-isomorphic case
        belongs_to  = lambda x, k: ((k <= resolution*x) & (resolution*x < k+1))
        is_boundary = lambda x, k: ((k == resolution-1) & (x == 1))
        return np.stack([ 
            (belongs_to(rhos, k) | is_boundary(rhos, k)) for k in range(resolution)
        ], 2)
    else: # altered isomorphic case
        if resolution % 2 == 0:
            raise ValueError("Resolution must be an odd number if iso=True.")
        belongs_to_lower = lambda x, k: ((k < resolution/2) & (x >= k/resolution) & (x < (k+1)/resolution))
        belongs_to_middle = lambda x, k: ((k == (resolution-1)/2)  & (x >= k/resolution) & (x <= (k+1)/resolution))
        belongs_to_upper = lambda x, k: ((k >resolution/2) & (x > k/resolution) & (x <= (k+1)/resolution))
        return np.stack([ 
            (belongs_to_lower(rhos,k) | belongs_to_middle(rhos,k) | belongs_to_upper(rhos,k))for k in range(resolution)
        ], 2)

def _random_ones_arrays(size: int, num_arrays: int, ones_per_array: int) -> NDArray:
    """
    Generates N arrays of given size, each containing a specified number of ones 
    at unique positions per row, ensuring rows are not identical.
    NOTE: made with ChatGPT

    Parameters
    ----------
    size : int
        The size of each array.
    num_arrays : int
        The number of arrays to generate.
    ones_per_array : int
        The number of ones per array.

    Returns
    -------
    numpy.ndarray
        A NumPy array of shape (num_arrays, size) where each row has exactly `ones_per_array` ones
        at unique positions per row, ensuring rows are unique.
    """
    if ones_per_array > size:
        raise ValueError("ones_per_array cannot exceed the array size (not enough space).")
    if num_arrays > math.comb(size, ones_per_array):
        raise ValueError("Not enough unique combinations of ones available.")

    arr = np.zeros((num_arrays, size), dtype=int)  # Initialize all zeros
    unique_rows = set()

    for i in range(num_arrays):
        while True:
            indices = tuple(sorted(np.random.choice(size, ones_per_array, replace=False)))  # Unique positions as a tuple
            if indices not in unique_rows:
                unique_rows.add(indices)
                break  # Found a unique row

        arr[i, list(indices)] = 1  # Set ones at selected positions

    return arr

# %%
