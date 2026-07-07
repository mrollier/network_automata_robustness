"""Graph builder invariants."""

import random

import pytest

from llna.networks import create_2d_torus_lattice, watts_strogatz_rewire


@pytest.mark.parametrize("degree", [4, 8])
def test_torus_is_degree_regular_and_connected(degree):
    L = 6
    g = create_2d_torus_lattice(L, degree=degree)
    assert g.vcount() == L * L
    assert set(g.degree()) == {degree}
    assert g.is_connected()
    assert not any(g.is_loop()) and not any(g.is_multiple())


def test_torus_moore_neighbourhood_of_origin():
    L = 5
    g = create_2d_torus_lattice(L, degree=8)
    # node (x=0, y=0) has index 0; wrap-around Moore neighbours:
    expected = {
        1,
        L - 1,  # (±1, 0)
        L,
        L * (L - 1),  # (0, ±1)
        L + 1,
        L + L - 1,  # (±1, +1)
        L * (L - 1) + 1,
        L * L - 1,  # (±1, -1)
    }
    assert set(g.neighbors(0)) == expected


def test_torus_rejects_other_degrees():
    with pytest.raises(ValueError):
        create_2d_torus_lattice(4, degree=6)


def test_ws_rewire_preserves_counts_and_connectivity():
    g = create_2d_torus_lattice(6, degree=8)
    random.seed(7)
    rewired = watts_strogatz_rewire(g, 0.2)
    assert rewired.vcount() == g.vcount()
    assert rewired.ecount() == g.ecount()
    assert rewired.is_connected()
    assert rewired.get_edgelist() != g.get_edgelist()


def test_ws_rewire_is_deterministic_under_seed():
    g = create_2d_torus_lattice(5, degree=4)
    random.seed(123)
    first = watts_strogatz_rewire(g, 0.3).get_edgelist()
    random.seed(123)
    second = watts_strogatz_rewire(g, 0.3).get_edgelist()
    assert first == second


def test_ws_rewire_probability_zero_is_identity():
    g = create_2d_torus_lattice(5, degree=4)
    random.seed(1)
    assert watts_strogatz_rewire(g, 0.0).get_edgelist() == g.get_edgelist()
