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

# warnings and exceptions
import warnings

# particular packages
from src.automata import LLNA
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
    This is defined by the diameter of the subgraph of all affected nodes, divided by T.
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
        Defaults to True. If False, the defect diameters are not normalised (divided by T).

    Returns
    -------
    diameters : numpy.ndarray
        An array of length T with the defect diameters related to a defect in each of the N nodes.
        If norm==False, the diameters are not normalised (divided by T).
    """
    N = len(states)

    # get ID of edges (bidirectional)
    graph.to_directed()
    edges = tc.tensor(graph.get_edgelist()).T
    graph.to_undirected()

    # parallellise all possible defects
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
        diameters = diameters/(T+1)
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
        list of integers corresponding to the indices of the activated density intervals in the B set
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
    rho_intervals = _interval_encoding(resolution, rhos[np.newaxis,:], iso=True)[0].argmax(axis=1)
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
        list of integers corresponding to the indices of the activated density intervals in the B set

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
        list of integers corresponding to the indices of the activated density intervals in the B set

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
    norm_degree:bool=True,
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
        list of integers corresponding to the indices of the activated density intervals in the B set
    degree : int
        The degree of the node you want to calculate the Hamming weight for
    norm : bool
        True by default. Maps the Hamming weight to a value from 0 to 1. 
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
        # find all possible densities except density 1 (which cannot increase)
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
        # add a zero at the end and return
        return np.append(0, NS_dec)
    
    # overall neighbourhood sensitivity
    def _nbh_sens(resolution, B_set, S_set, si, degree, iso=True):
        sum_values = np.linspace(0, degree, degree+1)
        NS_dec = _nbh_sens_dec(resolution, B_set, S_set, si, degree, iso=iso)
        NS_inc = _nbh_sens_inc(resolution, B_set, S_set, si, degree, iso=iso)
        return sum_values*NS_dec + (degree-sum_values)*NS_inc
    
    # use functions above to find BS
    prefactor = 2**(-degree-1)
    if norm_degree:
        prefactor /= (degree+1)
    summation = 0
    configs_per_rho = np.array([comb(degree, k, exact=False) for k in range(degree + 1)])
    for si in [0,1]:
        IS = _id_sens(resolution, B_set, S_set, degree, iso=iso)
        NS = _nbh_sens(resolution, B_set, S_set, si, degree, iso=iso)
        term = np.sum((IS + NS) * configs_per_rho)
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
        list of integers corresponding to the indices of the activated density intervals in the B set
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

# Define a function that identifies equivalent update rule
def return_equivalent_rule(
    resolution:int,
    B_set:Union[NDArray[np.int_], Sequence[int]],
    S_set:Union[NDArray[np.int_], Sequence[int]],
    return_decimals:bool=False
) -> Union[Tuple[NDArray[np.int_], NDArray[np.int_]], Tuple[int,int]]:
    """
    A function that returns the beta and sigma integer of the equivalent local update rule.

    Parameters
    ----------
    resolution : int
        Positive integer indicating the resolution of the local update rule.
    B_set : list or numpy.ndarray
        list of integers corresponding to the indices of the activated density intervals in the B set.
    S_set : list or numpy.ndarray
        list of integers corresponding to the indices of the activated density intervals in the B set.

    Returns
    -------
    B_set_equiv : list or numpy.ndarray
        list of integers corresponding to the indices of the activated density intervals in the B set of the equivalent rule.
    S_set_equiv : list or numpy.ndarray
        list of integers corresponding to the indices of the activated density intervals in the B set of the equivalent rule.
    """
    # make sure the sets are numpy.ndarray
    B_set = np.asarray(B_set, dtype=int)
    S_set = np.asarray(S_set, dtype=int)
    # mirror the sets
    B_set_mirror = resolution - 1 - B_set
    S_set_mirror = resolution - 1 - S_set
    # complement the sets
    B_set_mirror_comp = np.setdiff1d(np.arange(resolution), B_set_mirror)
    S_set_mirror_comp = np.setdiff1d(np.arange(resolution), S_set_mirror)
    # switch up the sets
    B_set_equiv = S_set_mirror_comp
    S_set_equiv = B_set_mirror_comp
    if not return_decimals:
        return B_set_equiv, S_set_equiv
    beta_equiv = int(np.sum([2**idx for idx in B_set_equiv]))
    sigma_equiv = int(np.sum([2**idx for idx in S_set_equiv]))
    return beta_equiv, sigma_equiv

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
    triplet = J[1,0:3]
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

def binary_indices(n: int) -> list[int]:
    """
    Returns a list of indices where bits are 1 in the binary representation of n.
    """
    return [i for i, bit in enumerate(bin(n)[:1:-1]) if bit == '1']

def eca_as_binary(eca:int) -> str:
    """
    Returns the ECA as a binary number consisting of exactly 8 digits.
    """
    # check value of ECA
    if not _is_eca(eca):
        raise ValueError(f"The integer {eca} is not a valid ECA.")
    # body
    eca_binary = np.base_repr(eca, base=2)
    eca_binary = '0'*(8-len(eca_binary)) + eca_binary
    return eca_binary

def lr_symmetric_eca(eca:int) -> int:
    """
    Calculates the ECA that is left-right symmetric to the ECA in the argument

    Parameters
    ----------
    eca : int
        A Wolfram rule, i.e. a natural number smaller than 256.
    
    Returns
    -------
    lr_eca : int
        Another Wolfram rule, that is left-right symmetric to the one in the argument
    """
    # check value of ECA
    if not _is_eca(eca):
        raise ValueError(f"The integer {eca} is not a valid ECA.")
    # body
    eca_bin = eca_as_binary(eca)
    lr_eca_bin = eca_bin[0] + eca_bin[4] + eca_bin[2] + eca_bin[6] + eca_bin[1] + eca_bin[5] + eca_bin[3] + eca_bin[7]
    lr_eca = int(lr_eca_bin, base=2)
    return lr_eca

def bw_symmetric_eca(eca:int) -> int:
    """
    Calculates the ECA that is black-white symmetric to the ECA in the argument

    Parameters
    ----------
    eca : int
        A Wolfram rule, i.e. a natural number smaller than 256.
    
    Returns
    -------
    lr_eca : int
        Another Wolfram rule, that is left-right symmetric to the one in the argument
    """
    # check value of ECA
    if not _is_eca(eca):
        raise ValueError(f"The integer {eca} is not a valid ECA.")
    # body
    eca_bin = eca_as_binary(eca)
    inv_eca_bin = eca_bin[::-1]
    bw_eca = 255 - int(inv_eca_bin, base=2)
    return bw_eca

def lp_class_dict() -> dict:
    lp_dict = { "null": [0, 8, 32, 40, 64, 96, 128, 136, 160, 168, 192, 224, 234, 235, 238, 239, 248, 249, 250, 251, 252, 253, 254, 255], \
                "fixed point": [2, 4, 10, 12, 13, 16, 24, 34, 36, 42, 44, 46, 48, 56, 57, 58, 66, 68, 69, 72, 76, 77, 78, 79, 80, 92, 93, 98, 99, 100, 104, 112, 114, 116, 130, 132, 138, 139, 140, 141, 144, 152, 162, 163, 164, 170, 171, 172, 174, 175, 176, 177, 184, 185, 186, 187, 188, 189, 190, 191, 194, 196, 197, 200, 202, 203, 204, 205, 206, 207, 208, 209, 216, 217, 218, 219, 220, 221, 222, 223, 226, 227, 228, 230, 231, 232, 233, 236, 237, 240, 241, 242, 243, 244, 245, 246, 247], \
                "periodic": [1, 3, 5, 6, 7, 9, 11, 14, 15, 17, 19, 20, 21, 23, 25, 27, 28, 29, 31, 33, 35, 37, 38, 39, 41, 43, 47, 49, 50, 51, 52, 53, 55, 59, 61, 62, 63, 65, 67, 70, 71, 74, 81, 83, 84, 85, 87, 88, 91, 94, 95, 97, 103, 107, 108, 111, 113, 115, 117, 118, 119, 121, 123, 125, 127, 131, 133, 134, 142, 143, 145, 148, 155, 156, 157, 158, 159, 173, 178, 179, 198, 199, 201, 211, 212, 213, 214, 215, 229], \
                "locally chaotic": [26, 73, 82, 109, 154, 166, 167, 180, 181, 210], \
                "chaotic": [18, 22, 30, 45, 54, 60, 75, 86, 89, 90, 101, 102, 105, 106, 110, 120, 122, 124, 126, 129, 135, 137, 146, 147, 149, 150, 151, 153, 161, 165, 169, 182, 183, 193, 195, 225]}
    return lp_dict

def eca_to_llna(eca:int):
    if not eca_is_llna(eca):
        raise ValueError(f"Rule {eca} cannot be translated to an LLNA.")
    eca_binary_reverse = eca_as_binary(eca)[::-1]
    B_set = []
    S_set = []
    for interval, dead_bit in enumerate([0, 1, 5]):
        if eca_binary_reverse[dead_bit]=='1':
            B_set.append(interval)
    for interval, live_bit in enumerate([2, 3, 7]):
        if eca_binary_reverse[live_bit]=='1':
            S_set.append(interval)
    return B_set, S_set

def eca_is_totalistic(eca:int) -> bool:
    if not _is_eca(eca):
        raise ValueError(f"ECA '{eca}' is not recognised as an elementary cellular automaton. Choose an integer from 0 to 255.")
    # make list of all totalistic ECAs
    totalistic_ecas = []
    for b7 in [0,1]:
        for b6 in [0,1]:
            b5 = b6; b3 = b6
            for b4 in [0,1]:
                b2 = b4; b1=b4
                for b0 in [0,1]:
                    tot_eca = 2**7*b7 + 2**6*b6 + 2**5*b5 + 2**4*b4 + 2**3*b3 + 2**2*b2 + 2**1*b1 + 2**0*b0
                    totalistic_ecas.append(tot_eca)
    if eca in totalistic_ecas:
        return True
    return False

def eca_is_llna(eca:int) -> bool:
    if not _is_eca(eca):
        raise ValueError(f"ECA '{eca}' is not recognised as an elementary cellular automaton. Choose an integer from 0 to 255.")
    # make list of all totalistic ECAs
    llna_ecas = []
    for b7 in [0,1]:
        for b6 in [0,1]:
            b3 = b6
            for b5 in [0,1]:
                for b4 in [0,1]:
                    b1=b4
                    for b2 in [0,1]:
                        for b0 in [0,1]:
                            llna_eca = 2**7*b7 + 2**6*b6 + 2**5*b5 + 2**4*b4 + 2**3*b3 + 2**2*b2 + 2**1*b1 + 2**0*b0
                            llna_ecas.append(llna_eca)
    if eca in llna_ecas:
        return True
    return False

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
    x_binned.insert(0, 0)  # Insert x = 0 at the beginning
    y_binned.insert(0, 0)  # Insert y = 0 at the beginning

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
    if not _is_eca(eca):
        raise Exception(f"ECA '{eca}' is not recognised as an elementary cellular automaton. Choose an integer from 0 to 255.")
    # make list of all constant-Jacobian ECAs
    # constantJ_ecas = []
    # for b7 in [0,1]:
    #     b5 = b7
    #     for b6 in [0,1]:
    #         b4 = b6
    #         for b3 in [0,1]:
    #             b1 = b3
    #             for b2 in [0,1]:
    #                 b0 = b2
    #                 constantJ_eca = 2**7*b7 + 2**6*b6 + 2**5*b5 + 2**4*b4 + 2**3*b3 + 2**2*b2 + 2**1*b1 + 2**0*b0
    #                 constantJ_ecas.append(constantJ_eca)
    # for b7 in [0,1]:
    #     b5 = 1-b7
    #     for b6 in [0,1]:
    #         b4 = 1-b6
    #         for b3 in [0,1]:
    #             b1 = 1-b3
    #             for b2 in [0,1]:
    #                 b0 = 1-b2
    #                 constantJ_eca = 2**7*b7 + 2**6*b6 + 2**5*b5 + 2**4*b4 + 2**3*b3 + 2**2*b2 + 2**1*b1 + 2**0*b0
    #                 constantJ_ecas.append(constantJ_eca)
    # if eca in constantJ_ecas:
    #     return True
    # return False
    all_constantJ_rules = [0, 255, 85, 170, 51, 204, 102, 153, 15, 240, 90, 165, 60, 195, 105, 150]
    if eca not in all_constantJ_rules:
        return False
    return True


#%% helper functions
def _is_eca(eca:int):
    # Check whether eca is a valid integer in [0, 255]
    if eca < 0:
        return False
    if eca > 255:
        return False
    return True

def _interval_encoding(resolution:int, rhos, iso=True):
    if not iso: # classic psuedo-isomorphic case
        belongs_to  = lambda x, k: ((k <= resolution*x) & (resolution*x < k+1))
        is_boundary = lambda x, k: ((k == resolution-1) & (x == 1))
        return np.stack([ 
            (belongs_to(rhos, k) | is_boundary(rhos, k)) for k in range(resolution)
        ], 2)
    else: # altered isomorphic case
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
