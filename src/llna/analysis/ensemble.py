"""Ensemble statistics and initial-configuration helpers."""

import math

import numpy as np
from numpy.typing import NDArray


def median_and_percentiles_over_ensemble(arrays, delta_t, lower_percentile=0.25, upper_percentile=0.75,
                                         time_axis=0):
    """
    Find the convergence value of an ensemble of time series: pool the final
    `delta_t` timesteps of every ensemble member and return median + quantiles.

    Parameters
    ----------
    arrays : numpy.ndarray
        2D array of time series. Time runs along `time_axis` (default 0,
        i.e. shape [T, ensemble]); pass time_axis=1 for [ensemble, T] input.
    delta_t : int
        Number of final timesteps considered converged.
    """
    if arrays.ndim != 2:
        raise ValueError(f"The input array has {arrays.ndim} dimensions instead of 2.")
    if (lower_percentile > 0.5) or (upper_percentile < 0.5):
        raise ValueError("The median is not within the percentiles.")
    if time_axis not in (0, 1):
        raise ValueError(f"time_axis must be 0 or 1, got {time_axis}.")
    if time_axis == 1:
        arrays = arrays.T
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

def init_config_with_dens(N, dens, rng=None):
    """
    Initialize a configuration array with a given density of ones.

    Parameters:
    N (int): The size of the array.
    dens (float): The density of ones in the array (between 0 and 1).
    rng (numpy.random.Generator, optional): source of randomness. Defaults to
        the global numpy RNG (reproducible via np.random.seed).

    Returns:
    numpy.ndarray: A shuffled array of size N with the specified density of ones.
    """
    s0 = np.zeros(N, dtype=int)
    s0[:np.round(dens*N).astype(int)] = 1.
    if rng is None:
        np.random.shuffle(s0)
    else:
        rng.shuffle(s0)
    return s0

#%% helper functions

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

