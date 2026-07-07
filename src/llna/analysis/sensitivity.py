"""Genotype metrics of a local update rule: Hamming weight and sensitivities."""

from typing import Sequence, Union

import igraph as ig
import numpy as np
from numpy.typing import NDArray
from scipy.special import comb
from scipy.stats import binom

from llna.analysis._encoding import _interval_encoding


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

