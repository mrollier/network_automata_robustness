"""Jacobian correctness against brute-force single-bit flips."""

import igraph as ig
import numpy as np
import torch as tc

from llna.analysis import jacobian
from llna.automata import LLNA
from llna.networks import create_2d_torus_lattice


def brute_force_jacobian(graph: ig.Graph, model: LLNA, states: np.ndarray) -> np.ndarray:
    g = graph.copy()
    g.to_directed()
    E = tc.tensor(g.get_edgelist(), dtype=tc.long).T
    base = model.step(E, tc.tensor(states[np.newaxis, :].astype(np.int64))).numpy()[0]
    N = len(states)
    J = np.zeros((N, N), dtype=int)
    for j in range(N):
        flipped = states.copy()
        flipped[j] ^= 1
        nxt = model.step(E, tc.tensor(flipped[np.newaxis, :].astype(np.int64))).numpy()[0]
        J[:, j] = (nxt != base).astype(int)
    return J


def test_jacobian_matches_brute_force_torus():
    graph = create_2d_torus_lattice(4, degree=8)
    model = LLNA(9, [3], [2, 3], iso=True)  # Game of Life
    rng = np.random.default_rng(1)
    for _ in range(3):
        states = rng.integers(0, 2, graph.vcount())
        assert np.array_equal(jacobian(graph, model, states), brute_force_jacobian(graph, model, states))


def test_jacobian_matches_brute_force_er():
    import random

    random.seed(4)
    while True:
        graph = ig.Graph.Erdos_Renyi(n=18, p=0.25)
        if graph.is_connected():
            break
    model = LLNA(5, [1, 2, 3], [0, 4], iso=True)
    rng = np.random.default_rng(2)
    states = rng.integers(0, 2, graph.vcount())
    assert np.array_equal(jacobian(graph, model, states), brute_force_jacobian(graph, model, states))


def test_jacobian_return_next_evolves_state():
    graph = create_2d_torus_lattice(4, degree=4)
    model = LLNA(2, [1], [0, 1], iso=False)
    rng = np.random.default_rng(3)
    states = rng.integers(0, 2, graph.vcount())
    g = graph.copy()
    g.to_directed()
    E = tc.tensor(g.get_edgelist(), dtype=tc.long).T
    _, states_next = jacobian(graph, model, states, return_next=True)
    expected = model.step(E, tc.tensor(states[np.newaxis, :].astype(np.int64))).numpy()[0]
    assert np.array_equal(states_next, expected.astype(int))
