from __future__ import annotations
import numpy as np
from typing import Union, Tuple, Sequence
from numpy.typing import NDArray
from tqdm import tqdm

def return_life_like_dict():
    """
    Returns a dictionary with then name of the life-like rule and the (beta, sigma) couple for resolution 9.
    List copied from Wikipedia.
    """
    life_like_dict = {"replicator" : (170, 170),
                      "seeds" : (4, 0),
                      "no_name" : (36, 16),
                      "life_without_death" : (8, 511),
                      "life" : (8, 12),
                      "34_life" : (24, 24),
                      "diamoeba" : (488, 480),
                      "2x2" : (72, 38),
                      "highlife" : (72, 12),
                      "day&night" : (456, 472),
                      "morley" : (328, 52),
                      "anneal" : (464, 488)}
    return life_like_dict

def get_nonequiv_rules(resolution:int, return_self_equiv=False):
    # takes a long time for large resolutions. Some of these lists have been saved elsewhere for easy loading.
    betas = range(2**resolution)
    sigmas = range(2**resolution)
    equiv_rule_list = []
    nonequiv_rule_list = []
    self_equiv_rule_list = []
    for beta in tqdm(betas, total=2**resolution):
        born_if = binary_indices(beta)
        for sigma in sigmas:
            if (beta, sigma) not in equiv_rule_list:
                born_if = binary_indices(beta)
                survive_if = binary_indices(sigma)
                # add the equivalent rule to the list
                beta_equiv, sigma_equiv = return_equivalent_rule(resolution, born_if, survive_if, return_decimals=True)
                nonequiv_rule_list.append((beta, sigma))
                if (beta, sigma) != (beta_equiv, sigma_equiv):
                    equiv_rule_list.append((beta_equiv, sigma_equiv))
                else:
                    self_equiv_rule_list.append((beta, sigma))
    if return_self_equiv:
        return nonequiv_rule_list, self_equiv_rule_list
    return nonequiv_rule_list

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
        list of integers corresponding to the indices of the activated density intervals in the S set.

    Returns
    -------
    B_set_equiv : list or numpy.ndarray
        list of integers corresponding to the indices of the activated density intervals in the B set of the equivalent rule.
    S_set_equiv : list or numpy.ndarray
        list of integers corresponding to the indices of the activated density intervals in the S set of the equivalent rule.
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
    # TODO: make a function that looks for outer-totalistic ECAs
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


#%% helper functions

def _is_eca(eca:int):
    # Check whether eca is a valid integer in [0, 255]
    if eca < 0:
        return False
    if eca > 255:
        return False
    return True