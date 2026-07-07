"""Semantic tests of the LLNA dynamics against an independent reference."""

import numpy as np
import pytest
import torch as tc

from llna.automata import LLNA
from llna.networks import create_2d_torus_lattice


def gol_reference_step(grid: np.ndarray) -> np.ndarray:
    """Straightforward numpy Game of Life on a torus (Moore neighbourhood)."""
    neighbours = sum(
        np.roll(np.roll(grid, dy, axis=0), dx, axis=1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if (dy, dx) != (0, 0)
    )
    born = (grid == 0) & (neighbours == 3)
    survive = (grid == 1) & ((neighbours == 2) | (neighbours == 3))
    return (born | survive).astype(np.int64)


def edge_tensor(graph) -> tc.Tensor:
    g = graph.copy()
    g.to_directed()
    return tc.tensor(g.get_edgelist(), dtype=tc.long).T


@pytest.fixture(scope="module")
def gol_setup():
    L = 12
    graph = create_2d_torus_lattice(L, degree=8)
    # Game of Life as an LLNA: B{3}/S{2,3} at resolution 9 (beta=8, sigma=12);
    # on a degree-8 torus, interval k contains exactly the density k/8.
    model = LLNA(9, [3], [2, 3], iso=True)
    return L, edge_tensor(graph), model


def test_llna_reproduces_game_of_life_soup(gol_setup):
    L, E, model = gol_setup
    rng = np.random.default_rng(42)
    grid = (rng.random((L, L)) < 0.4).astype(np.int64)
    traj = model.forward(E, tc.tensor(grid.reshape(1, -1)), T=20).numpy()[0]
    for t in range(1, 21):
        grid = gol_reference_step(grid)
        assert np.array_equal(traj[t].reshape(L, L), grid), f"diverged at step {t}"


def test_llna_glider_translates(gol_setup):
    L, E, model = gol_setup
    grid = np.zeros((L, L), dtype=np.int64)
    # standard glider
    for y, x in [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)]:
        grid[y, x] = 1
    traj = model.forward(E, tc.tensor(grid.reshape(1, -1)), T=4).numpy()[0]
    final = traj[4].reshape(L, L)
    expected = np.roll(np.roll(grid, 1, axis=0), 1, axis=1)  # glider moves (+1, +1) per period
    assert np.array_equal(final, expected)
    assert final.sum() == 5


def test_rule_roundtrip_and_str():
    model = LLNA(9, [3], [2, 3], iso=True)
    assert model.rule == ([3], [2, 3])
    assert str(model) == "R9B8S12"
    assert model.resolution == 9


def test_empty_rule_kills_everything():
    graph = create_2d_torus_lattice(4, degree=4)
    model = LLNA(2, [], [], iso=False)
    configs = tc.tensor(np.eye(16, dtype=np.int64)[:5])
    out = model.step(edge_tensor(graph), configs)
    assert tc.all(out == 0)


def test_iso_requires_odd_resolution():
    with pytest.raises(ValueError):
        LLNA(4, [1], [2], iso=True)
