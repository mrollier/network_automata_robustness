# %% import packages

# regular packages
import numpy as np
import torch as tc
import igraph as ig
from typing import Union, Tuple

# particular packages
from src.automata import LLNA

# %% utilitary functions
def jacobian(graph:ig.Graph, model:LLNA, states:np.ndarray, return_next:bool=False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
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

def defect_diameter(graph:ig.Graph, model:LLNA, states:np.ndarray, T:int, norm:bool=True) -> np.ndarray:
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
        subG = G.subgraph(delta_nodes)
        diameter = subG.diameter()
        diameters += [diameter]
    diameters = np.array(diameters)
    if norm:
        diameter /= T
    return diameters