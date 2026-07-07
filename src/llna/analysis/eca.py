"""Elementary-cellular-automaton reference dynamics (isolates cellpylib)."""

import cellpylib as cpl
import numpy as np
from numpy.typing import NDArray

from llna.rules import _is_eca


def jacobian_ECA(rule: int, states: NDArray, return_next: bool) -> NDArray | tuple[NDArray, NDArray]:
    """
    TODO: add description
    """
    # number of nodes
    N = len(states)
    # calculate unperturbed next-timestep state vector
    states_next = cpl.evolve(
        states[np.newaxis, :], timesteps=2, apply_rule=lambda n, c, t: cpl.nks_rule(n, rule)
    )[-1]
    states_all_next = np.tile(states_next, (N, 1))

    # make matrix with all possible single-defect perturbations
    states_defect_all = (np.tile(states, (N, 1)) + np.diag(np.ones(N, dtype=int))) % 2
    # calculate perturbed next-timestep state vector (N times in parallel)
    states_defect_all_next = np.empty_like(states_defect_all)
    for tt, states_defect in enumerate(states_defect_all):
        states_defect_next = cpl.evolve(
            states_defect[np.newaxis, :], timesteps=2, apply_rule=lambda n, c, t: cpl.nks_rule(n, rule)
        )[-1]
        states_defect_all_next[tt] = states_defect_next

    # compile Jacobian with XOR operator (the Boolean derivative)
    J = (states_all_next + states_defect_all_next) % 2

    # return transpose (to match definition)
    if return_next:
        return J.T, states_next
    return J.T


def eca_has_constantJ(eca: int) -> bool:
    # TODO: this is hard-coded for now.
    if not _is_eca(eca):
        raise Exception(
            f"ECA '{eca}' is not recognised as an elementary cellular automaton. Choose an integer from 0 to 255."
        )
    all_constantJ_rules = [0, 255, 85, 170, 51, 204, 102, 153, 15, 240, 90, 165, 60, 195, 105, 150]
    if eca not in all_constantJ_rules:
        return False
    return True
