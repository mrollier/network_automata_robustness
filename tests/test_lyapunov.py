"""Semantic tests tying the numerical Lyapunov pipeline to closed forms."""

import igraph as ig
import numpy as np
import pytest

from llna.analysis import (
    calculate_Yt,
    jacobian,
    jacobian_ECA,
    lyapunov_spectrum,
    lyapunov_spectrum_analytical,
)
from llna.automata import LLNA
from llna.rules import eca_to_llna


def ring(n: int) -> ig.Graph:
    return ig.Graph.Ring(n, circular=True)


def test_numerical_spectrum_matches_analytical_rule150():
    """ECA 150 (parity) has a constant Jacobian; the numerical tangent-space
    spectrum on a ring must reproduce the circulant closed form."""
    N, T = 32, 6
    b_set, s_set = eca_to_llna(150)
    model = LLNA(3, b_set, s_set, iso=False)
    rng = np.random.default_rng(3)
    states = rng.integers(0, 2, N)
    Yt = calculate_Yt(ring(N), model, states, T=T)
    with np.errstate(divide="ignore"):
        numerical = lyapunov_spectrum(Yt, T=T)
    analytical = lyapunov_spectrum_analytical(150, N)
    finite = np.sort(numerical[np.isfinite(numerical)])
    assert np.allclose(finite, np.sort(analytical), atol=1e-8)


def test_jacobian_of_parity_rule_is_adjacency_without_self():
    """For the parity-style LLNA of ECA 150, every neighbour flip flips the
    node, and a self flip flips the node: J = A + I on any state."""
    N = 16
    g = ring(N)
    b_set, s_set = eca_to_llna(150)
    model = LLNA(3, b_set, s_set, iso=False)
    rng = np.random.default_rng(5)
    states = rng.integers(0, 2, N)
    J = jacobian(g, model, states)
    A = np.array(g.get_adjacency().data)
    assert np.array_equal(J, A + np.eye(N, dtype=int))


def test_jacobian_ECA_rule150_is_tridiagonal_circulant():
    N = 12
    states = np.zeros(N)
    J = jacobian_ECA(150, states, return_next=False)
    expected = (np.eye(N, k=0) + np.eye(N, k=1) + np.eye(N, k=-1) + np.eye(N, k=N - 1) + np.eye(N, k=-(N - 1))).astype(int)
    assert np.array_equal(J, expected)


def test_analytical_spectrum_mle_is_log3_for_rule150():
    # max_k |1 + 2cos(2πk/N)| is attained at k=0 and equals exactly 3.
    vals = lyapunov_spectrum_analytical(150, 30)
    assert np.isclose(vals.max(), np.log(3.0))


def test_analytical_spectrum_never_returns_nan():
    """When N is divisible by 3 the circulant symbol of rules 150/105 has an
    exact zero; rounding puts the squared singular value at -epsilon. The NaN
    from sqrt(-eps) is currently masked by the >0 filter."""
    for rule in [150, 105]:
        for N in [3, 6, 9, 63, 64]:
            vals = lyapunov_spectrum_analytical(rule, N)
            assert not np.isnan(vals).any()


def test_analytical_spectrum_emits_no_warning():
    """Eigenvalues are squared magnitudes of the circulant symbol, hence >= 0
    mathematically; -epsilon rounding is clamped instead of reaching sqrt."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        lyapunov_spectrum_analytical(150, 6)
        lyapunov_spectrum_analytical(105, 63)
