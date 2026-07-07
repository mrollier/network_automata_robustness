"""Mean-field density propagation of a local update rule."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.special import comb
from scipy.stats import binom

from llna.analysis._encoding import _interval_encoding


def mean_field_dens_propagation(
    resolution: int,
    B_set: NDArray[np.int_] | Sequence[int],
    S_set: NDArray[np.int_] | Sequence[int],
    current_dens: float | NDArray,
    degree: int,
    iso: bool = True,
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
        raise ValueError(
            f"Degree {degree} not allowed. The degree must be a non-zero natural number smaller than 1023."
        )
    if iso and resolution % 2 == 0:
        raise ValueError(
            f"Resolution {resolution} is not possible when iso=True. Choose an odd positive integer."
        )
    if (B_set and (np.max(B_set) >= resolution)) or (S_set and (np.max(S_set) >= resolution)):
        raise ValueError(f"Resolution {resolution} is to small for the provided update intervals.")
    current_dens = np.atleast_1d(current_dens)
    if (np.min(current_dens) < 0) or (np.max(current_dens) > 1):
        raise ValueError(
            f"Average state density must be between 0 and 1. Now the maximum is {np.max(current_dens)} and the minimum is {np.max(current_dens)}."
        )

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
        raise ValueError(
            f"Degree {degree} not allowed. The degree must be a non-zero natural number smaller than 1023."
        )
    if iso and resolution % 2 == 0:
        raise ValueError(
            f"Resolution {resolution} is not possible when iso=True. Choose an odd positive integer."
        )
    if (B_set and (np.max(B_set) >= resolution)) or (S_set and (np.max(S_set) >= resolution)):
        raise ValueError(f"Resolution {resolution} is to small for the provided update intervals.")
    if (rho_star < 0) or (rho_star > 1):
        raise ValueError(f"Equilibrium state density `rho_star` must be between 0 and (not {rho_star}).")

    # find all the possible rho_i values for this degree
    rhos = np.linspace(0, 1, degree + 1)
    # check in which interval they are situated
    rho_intervals = _interval_encoding(resolution, rhos[np.newaxis, :], iso=iso)[0].argmax(axis=1)
    # find binomium elements
    binomium_factor = np.array([comb(degree, q) for q in range(0, degree + 1)])
    # find what this outputs to
    born_truthtable = np.array([int(rho in B_set) for rho in rho_intervals])
    survive_truthtable = np.array([int(rho in S_set) for rho in rho_intervals])
    # calculate the binomial probabilities for various sums q, based on the current average density
    sum_prefactor_born = np.array(
        [
            rho_star ** (q - 1) * (1 - rho_star) ** (degree - q) * (q - rho_star * (1 + degree))
            for q in range(degree + 1)
        ]
    )
    sum_prefactor_survive = np.array(
        [
            rho_star**q * (1 - rho_star) ** (degree - q - 1) * (1 + q - rho_star * (1 + degree))
            for q in range(degree + 1)
        ]
    )
    # calculate the weighted born and survive truthtables (based on probability of finding the central node alive)
    born_truthtable_weighted = born_truthtable * sum_prefactor_born
    survive_truthtable_weighted = survive_truthtable * sum_prefactor_survive
    # sum over all possible sum values
    slope = np.sum(binomium_factor * (born_truthtable_weighted + survive_truthtable_weighted))
    # return the next density and make sure it's between 0 and 1
    return slope
