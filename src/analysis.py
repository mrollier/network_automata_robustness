# %% import packages

# regular packages
import numpy as np
import torch as tc
import igraph as ig
# particular packages
from src.automata import LLNA

# %% utilitary functions
def jacobian(graph:ig.Graph, model:LLNA, states:np.ndarray, return_next:bool=False) -> np.ndarray:
    """
    Returns the Jacobian matrix of partial Boolean derivatives. J[i,j] can be either 0 or 1,
    and answers the question 'is node i affected by a change in the state of node j in the previous time step?'
    If it is, J[i,j] is 1. Otherwise, J[i,j] is 0.
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

    # make matrix with all possible single-defect perturbations
    states_defect_all = (np.tile(states, (N,1)) + np.diag(np.ones(N, dtype=int))) % 2
    # calculate perturbed next-timestep state vector (N times in parallel)
    states_defect_all_next = np.array(model.step(edges, tc.tensor(states_defect_all)), dtype=int)

    # compile Jacobian with XOR operator (the Boolean derivative)
    J = (states_all_next + states_defect_all_next) % 2

    # return transpose (to match definition)
    if return_next:
        return J.T, states_all_next[0]
    return J.T
