"""End-to-end CLI sweep on tiny parameters + SweepFile resume semantics."""

import subprocess
import sys

import h5py
import numpy as np
import pytest
import torch as tc

from llna.automata import LLNA
from llna.cli import main
from llna.io import SweepFile
from llna.rules import binary_indices, get_nonequiv_rules
from llna.sweep import evolve_defect_pairs


def run_sweep(tmp_path, extra=()):
    out = tmp_path / "results.h5"
    argv = [
        "sweep",
        "--network-type",
        "toroidal-lattice",
        "--resolution",
        "3",
        "--L",
        "5",
        "--degree",
        "4",
        "--num-graphs",
        "2",
        "--num-init-conf",
        "3",
        "--T",
        "12",
        "--delta-t",
        "4",
        "--seed",
        "11",
        "--output",
        str(out),
    ] + list(extra)
    assert main(argv) == 0
    return out


def test_sweep_end_to_end_schema_and_values(tmp_path):
    out = run_sweep(tmp_path)
    with h5py.File(out) as f:
        grp = f["resolution3/toroidal-lattice"]
        rules = grp["rules"][...]
        assert rules.shape == (36, 2)
        for kind in ("state", "defect"):
            for stat in ("median", "q1", "q3"):
                assert grp[kind][stat].shape == (36,)
        assert grp.attrs["rows_done"] == 36
        assert grp.attrs["seed"] == 11
        assert "git_commit" in grp.attrs
        state_median = grp["state/median"][...]
        defect_median = grp["defect/median"][...]
    assert ((state_median >= 0) & (state_median <= 1)).all()
    assert ((defect_median >= 0) & (defect_median <= 1)).all()
    # the all-dead rule (beta=0, sigma=0) must converge to state 0, defect 0
    idx = int(np.where((rules == [0, 0]).all(axis=1))[0][0])
    assert state_median[idx] == 0.0 and defect_median[idx] == 0.0


def test_sweep_is_seed_reproducible(tmp_path):
    dir_a, dir_b = tmp_path / "a", tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()
    out_a, out_b = run_sweep(dir_a), run_sweep(dir_b)
    with h5py.File(out_a) as fa, h5py.File(out_b) as fb:
        ga, gb = fa["resolution3/toroidal-lattice"], fb["resolution3/toroidal-lattice"]
        for name in ("state/median", "defect/median", "state/q1", "defect/q3"):
            assert np.array_equal(ga[name][...], gb[name][...])


def test_sweep_chunked_resume_matches_single_run(tmp_path):
    whole = run_sweep(tmp_path)
    chunked_dir = tmp_path / "chunked"
    chunked_dir.mkdir()
    chunked = run_sweep(chunked_dir, extra=["--rule-chunk", "7"])
    with h5py.File(whole) as fw, h5py.File(chunked) as fc:
        gw, gc = fw["resolution3/toroidal-lattice"], fc["resolution3/toroidal-lattice"]
        for name in ("state/median", "state/q1", "state/q3", "defect/median", "defect/q1", "defect/q3"):
            assert np.array_equal(gw[name][...], gc[name][...])


def test_sweepfile_rejects_mismatched_rules(tmp_path):
    writer = SweepFile(tmp_path / "x.h5")
    rules = np.array([[0, 0], [1, 2]])
    writer.init_group("g", rules, {"state/median": ((), "f8")}, {"seed": 1})
    with pytest.raises(ValueError, match="different rule table"):
        writer.init_group("g", rules[::-1], {"state/median": ((), "f8")}, {"seed": 1})


def test_sweepfile_resume_point(tmp_path):
    writer = SweepFile(tmp_path / "x.h5")
    rules = np.arange(10).reshape(5, 2)
    cols = {"state/median": ((), "f8")}
    assert writer.init_group("g", rules, cols, {}) == 0
    writer.write_rows("g", 0, {"state/median": np.ones(3)})
    assert writer.init_group("g", rules, cols, {}) == 3
    writer.write_rows("g", 3, {"state/median": np.ones(2)})
    data, attrs = writer.read_group("g")
    assert attrs["rows_done"] == 5
    assert np.array_equal(data["state/median"], np.ones(5))


def test_defect_pairs_match_llna_reference():
    """evolve_defect_pairs equals per-rule LLNA forward + XOR reduction."""
    import igraph as ig

    graph = ig.Graph.Lattice([4, 4], circular=True)
    g = graph.copy()
    g.to_directed()
    E = tc.tensor(g.get_edgelist(), dtype=tc.long).T
    rng = np.random.default_rng(0)
    configs = (rng.random((4, 16)) < 0.5).astype(np.int64)
    defected = configs.copy()
    defected[np.arange(4), rng.choice(16, 4, replace=False)] ^= 1
    rules = get_nonequiv_rules(3)

    states, defects = evolve_defect_pairs(E, configs, defected, rules, 3, iso=True, T=8)
    for i, (beta, sigma) in enumerate(rules):
        model = LLNA(3, binary_indices(beta), binary_indices(sigma), iso=True)
        clean = model.forward(E, tc.tensor(configs), T=8).numpy()
        twin = model.forward(E, tc.tensor(defected), T=8).numpy()
        assert np.array_equal(states[i], clean.astype(np.float32).mean(axis=-1))
        assert np.array_equal(defects[i], (clean != twin).astype(np.float32).mean(axis=-1))


def test_console_entry_point_runs():
    result = subprocess.run(
        [sys.executable, "-m", "llna.cli", "rules", "--resolution", "3"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "36 non-equivalent rules" in result.stdout
