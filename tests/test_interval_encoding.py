"""Semantic contract of the density -> interval one-hot encodings."""

import numpy as np
import pytest
import torch as tc

from llna.analysis import _interval_encoding
from llna.automata import LLNA
from llna.engine import interval_index

CASES = [(2, False), (3, False), (3, True), (5, True), (9, False), (9, True)]


@pytest.mark.parametrize("res,iso", CASES)
def test_engine_interval_index_matches_onehot_argmax(res, iso):
    """The batched engine's integer index must agree with argmax of the
    torch one-hot on every density, including exact boundaries and their
    nearest float32 neighbours (the batched sweep's exactness depends on it)."""
    bounds = np.arange(res + 1, dtype=np.float32) / res
    grid = np.unique(
        np.clip(
            np.concatenate(
                [
                    np.linspace(0, 1, 2001, dtype=np.float32),
                    bounds,
                    np.nextafter(bounds, np.float32(-1.0)),
                    np.nextafter(bounds, np.float32(2.0)),
                ]
            ),
            0.0,
            1.0,
        )
    )
    p = tc.tensor(grid, dtype=tc.float32).unsqueeze(0)
    model = LLNA(res, [0], [0], iso=iso)
    onehot_argmax = model.interval_encoding(p).argmax(dim=2)
    assert tc.equal(interval_index(p, res, iso), onehot_argmax)


@pytest.mark.parametrize("res,iso", CASES)
def test_every_density_maps_to_exactly_one_interval(res, iso):
    grid = np.linspace(0, 1, 1001)
    model = LLNA(res, [0], [0], iso=iso)
    torch_onehot = model.interval_encoding(tc.tensor(grid, dtype=tc.float32).unsqueeze(0)).numpy()[0]
    numpy_onehot = _interval_encoding(res, grid[np.newaxis, :], iso=iso)[0]
    assert (torch_onehot.sum(axis=1) == 1).all()
    assert (numpy_onehot.sum(axis=1) == 1).all()


@pytest.mark.parametrize("res", [2, 3, 5, 9])
def test_noniso_boundaries_belong_to_upper_interval(res):
    """Classic encoding: [k/R, (k+1)/R) with rho=1 folded into the top interval."""
    for k in range(res):
        rho = np.array([[k / res]])
        assert _interval_encoding(res, rho, iso=False)[0, 0].argmax() == k
    assert _interval_encoding(res, np.array([[1.0]]), iso=False)[0, 0].argmax() == res - 1


@pytest.mark.parametrize("res", [3, 5, 9])
def test_iso_middle_interval_is_closed_on_both_sides(res):
    """Isomorphic encoding: the middle interval includes both of its edges."""
    mid = (res - 1) // 2
    lo, hi = mid / res, (mid + 1) / res
    enc = _interval_encoding(res, np.array([[lo, hi]]), iso=True)[0]
    assert enc[0].argmax() == mid
    assert enc[1].argmax() == mid


@pytest.mark.parametrize("res", [3, 5, 9])
def test_iso_encoding_is_symmetric_under_density_complement(res):
    """iso=True exists precisely so that rho -> 1-rho mirrors the intervals."""
    rng = np.random.default_rng(res)
    rhos = rng.random((1, 200))
    enc = _interval_encoding(res, rhos, iso=True)[0].argmax(axis=1)
    enc_flipped = _interval_encoding(res, 1.0 - rhos, iso=True)[0].argmax(axis=1)
    assert np.array_equal(enc_flipped, res - 1 - enc)
