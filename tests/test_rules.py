"""Semantic tests of the rule algebra."""

from pathlib import Path

import numpy as np
import pytest

from llna.rules import (
    binary_indices,
    bw_symmetric_eca,
    eca_is_llna,
    eca_to_llna,
    get_nonequiv_rules,
    lr_symmetric_eca,
    return_equivalent_rule,
    return_life_like_dict,
)

RULE_TABLE = Path(__file__).resolve().parents[1] / "data" / "rule_tables" / "all_nonequiv_res9_rules.npy"


def encode(indices) -> int:
    return int(sum(2**i for i in indices))


def test_binary_indices_roundtrip():
    for n in range(512):
        assert encode(binary_indices(n)) == n


@pytest.mark.parametrize("res", [2, 3, 5, 9])
def test_equivalence_is_an_involution(res):
    rng = np.random.default_rng(res)
    for _ in range(50):
        beta, sigma = rng.integers(0, 2**res, size=2)
        be, se = return_equivalent_rule(
            res, binary_indices(int(beta)), binary_indices(int(sigma)), return_decimals=True
        )
        b2, s2 = return_equivalent_rule(res, binary_indices(be), binary_indices(se), return_decimals=True)
        assert (b2, s2) == (int(beta), int(sigma))


@pytest.mark.parametrize("res,expected", [(2, 10), (3, 36), (5, 528)])
def test_nonequiv_count_matches_closed_form(res, expected):
    assert expected == (4**res + 2**res) // 2
    assert len(get_nonequiv_rules(res)) == expected


def test_nonequiv_representatives_are_lexicographic_minima():
    for res in [2, 3]:
        for beta, sigma in get_nonequiv_rules(res):
            be, se = return_equivalent_rule(
                res, binary_indices(beta), binary_indices(sigma), return_decimals=True
            )
            assert (beta, sigma) <= (be, se)


def test_res9_enumeration_reproduces_committed_table():
    """The vectorised enumeration must reproduce the committed R=9 table
    exactly, including order (the table was produced by the original
    loop-based implementation over several hours)."""
    table = np.load(RULE_TABLE)
    assert table.shape == ((4**9 + 2**9) // 2, 2)
    computed = np.array(get_nonequiv_rules(9), dtype=np.int64)
    assert np.array_equal(computed, table)


def test_load_nonequiv_rules_cache_roundtrip(tmp_path):
    from llna.rules import load_nonequiv_rules

    first = load_nonequiv_rules(3, cache_dir=tmp_path)
    assert (tmp_path / "all_nonequiv_res3_rules.npy").exists()
    second = load_nonequiv_rules(3, cache_dir=tmp_path)
    assert np.array_equal(first, second)
    assert np.array_equal(first, np.array(get_nonequiv_rules(3), dtype=np.int64))


def test_life_like_dict_encodings_valid():
    for name, (beta, sigma) in return_life_like_dict().items():
        assert 0 <= beta < 512 and 0 <= sigma < 512, name


def test_eca_symmetries_are_involutions():
    for eca in range(256):
        assert lr_symmetric_eca(lr_symmetric_eca(eca)) == eca
        assert bw_symmetric_eca(bw_symmetric_eca(eca)) == eca


def test_eca_to_llna_game_semantics():
    # ECA 150 (parity) as LLNA: born iff exactly one neighbour alive,
    # survive iff both or neither neighbour alive.
    assert eca_is_llna(150)
    b_set, s_set = eca_to_llna(150)
    assert (b_set, s_set) == ([1], [0, 2])
