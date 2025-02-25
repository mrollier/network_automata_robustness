# %% import packages

# regular packages
import numpy as np
import torch as tc
import igraph as ig
from typing import Union, Tuple

# warnings and exceptions
import warnings

# particular packages
from src.automata import LLNA
import cellpylib as cpl

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

def jacobian_ECA(rule:int, states:np.ndarray, return_next:bool) -> np.ndarray:
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
        subG = graph.subgraph(delta_nodes)
        diameter = subG.diameter()
        diameters += [diameter]
    diameters = np.array(diameters)
    if norm:
        diameters = diameters/(T+1)
    return diameters

def nbh_sensitivity(resolution: int, B_set:list, S_set:list) -> float:
    """
    Returns a value between 0 and 1, indicating how sensitive the output of this LLNA is to slightly changing the neighbourhood density.
    NOTE: this is a naive definition that does not take into account the degree distribution and the state density distribution.

    Parameters
    ----------
    model : src.automata.LLNA
        Life-like network automaton (custom class). Envelops the rules that govern the automaton.

    Returns
    -------
    nbh_sens : float
        Value ranging from 0 to 1, indicating low resp. high sensitivity.
    """
    B_set = np.array(B_set)
    S_set = np.array(S_set)
    # turn interval index into bits
    B_set_bin = np.zeros(resolution, dtype=int)
    S_set_bin = np.zeros(resolution, dtype=int)
    if B_set.size: B_set_bin[B_set] = 1
    if S_set.size: S_set_bin[S_set] = 1
    # perform XOR with shifted arrays
    borders_B = np.logical_xor(B_set_bin[1:], B_set_bin[:-1]).sum()
    borders_S = np.logical_xor(S_set_bin[1:], S_set_bin[:-1]).sum()
    # return total number of edges
    nbh_sens = (borders_B + borders_S)/(resolution-1)/2
    return nbh_sens

def id_sensitivity(resolution: int, B_set:list, S_set:list) -> float:
    """
    Returns a value between 0 and 1, indicating how sensitive the output of this LLNA is to changing the value of a particular node.
    NOTE: this is a naive definition that does not take into account the degree distribution and the state density distribution.

    Parameters
    ----------
    model : src.automata.LLNA
        Life-like network automaton (custom class). Envelops the rules that govern the automaton.

    Returns
    -------
    id_sens : float
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

def calculate_Yt(graph:ig.Graph, model:LLNA, states:np.ndarray, T:int) -> np.ndarray:
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
        Yt = np.matmul(J.astype(np.float64), Yt) # no mod 2!
    return Yt

def lyapunov_spectrum(Yt:np.ndarray, T:int) -> np.ndarray:
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

def lyapunov_spectrum_analytical(rule:int, N:int, return_finite_pct=False) -> np.ndarray:
    """
    Calculates the Lyapunov spectrum in the tangent-space interpretation, by taking the natural logarithm of the singular values of the evolved unit-perturbation sphere. Here we make use of the (supposed) fact that we are calculating the singular values for an ECA with a constant Jacobian. This Jacobian is circulant, which in turn allows for an analytical expression. In so doing, we avoid any numerical instabilities that tend to arise in the numerical approach.
    """
    # exceptions
    if not eca_has_constantJ(rule):
        raise Exception(f"Rule {rule} does not have a constant Jacobian.")
    if N<0:
        raise Exception(f"The number of nodes must be a natural number. Got N={N}.")
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
    singular_values = []
    for k in range(N):
        # note: for some reason this loop takes a long time
        arg = cc0 + 2*cc1*np.cos(2*k*np.pi/N) + 2*cc2*np.cos(4*k*np.pi/N)
        # numerical errors occur sometimes already here!
        if arg < 0: warnings.warn(f"Numerical issue: a square root argument is negative for rule {rule}: {arg}.", UserWarning)
        singular_value = np.sqrt(arg)
        singular_values.append(singular_value)
    singular_values = np.array(singular_values)
    # eliminate singular values of zero (and negative ones from numerical problems)
    nonzero_singular_values = singular_values[singular_values>0]
    lyapunov_values = np.log(nonzero_singular_values)
    if return_finite_pct:
        finite_pct = round(len(nonzero_singular_values) / N * 100)
        return lyapunov_values, finite_pct
    return lyapunov_values

def eca_as_binary(eca:int):
    """
    Returns the ECA as a binary number consisting of exactly 8 digits.
    """
    # check value of ECA
    if not _is_eca(eca):
        raise Exception(f"The integer {eca} is not a valid ECA.")
    # body
    eca_binary = np.base_repr(eca, base=2)
    eca_binary = '0'*(8-len(eca_binary)) + eca_binary
    return eca_binary

def lr_symmetric_eca(eca:int):
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
        raise Exception(f"The integer {eca} is not a valid ECA.")
    # body
    eca_bin = eca_as_binary(eca)
    lr_eca_bin = eca_bin[0] + eca_bin[4] + eca_bin[2] + eca_bin[6] + eca_bin[1] + eca_bin[5] + eca_bin[3] + eca_bin[7]
    lr_eca = int(lr_eca_bin, base=2)
    return lr_eca

def bw_symmetric_eca(eca:int):
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
        raise Exception(f"The integer {eca} is not a valid ECA.")
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
        raise Exception(f"Rule {eca} cannot be translated to an LLNA.")
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
        raise Exception(f"ECA '{eca}' is not recognised as an elementary cellular automaton. Choose an integer from 0 to 255.")
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
        raise Exception(f"ECA '{eca}' is not recognised as an elementary cellular automaton. Choose an integer from 0 to 255.")
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
# %%
