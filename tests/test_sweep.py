"""The batched sweep must reproduce single-rule LLNA trajectories exactly."""

import numpy as np
import pytest
import torch as tc

from llna.automata import LLNA
from llna.networks import create_2d_torus_lattice
from llna.rules import binary_indices, get_nonequiv_rules
from llna.sweep import evolve_rules_batch


@pytest.fixture(scope="module")
def setup():
    graph = create_2d_torus_lattice(6, degree=8)
    g = graph.copy()
    g.to_directed()
    E = tc.tensor(g.get_edgelist(), dtype=tc.long).T
    rng = np.random.default_rng(7)
    configs = (rng.random((5, graph.vcount())) < 0.5).astype(np.int64)
    return E, configs


def reference_trajectories(E, configs, rules, resolution, iso, T):
    out = []
    for beta, sigma in rules:
        model = LLNA(resolution, binary_indices(int(beta)), binary_indices(int(sigma)), iso=iso)
        traj = model.forward(E, tc.tensor(configs), T=T).numpy()
        out.append(traj.astype(np.uint8))
    return np.stack(out)


@pytest.mark.parametrize("resolution,iso", [(2, False), (3, True), (9, True)])
def test_batched_equals_single_rule(setup, resolution, iso):
    E, configs = setup
    if resolution == 9:
        rules = [(8, 12), (488, 464), (0, 0), (511, 511), (328, 52)]
    else:
        rules = get_nonequiv_rules(resolution)
    batched = evolve_rules_batch(E, configs, rules, resolution, iso=iso, T=15, reduce="none")
    reference = reference_trajectories(E, configs, rules, resolution, iso, T=15)
    assert np.array_equal(batched, reference)


def test_chunking_is_invariant(setup):
    E, configs = setup
    rules = get_nonequiv_rules(3)
    full = evolve_rules_batch(E, configs, rules, 3, iso=True, T=10, reduce="none")
    chunked = evolve_rules_batch(E, configs, rules, 3, iso=True, T=10, reduce="none", rule_chunk=7)
    assert np.array_equal(full, chunked)


def test_node_mean_matches_full_trajectory_mean(setup):
    E, configs = setup
    rules = [(8, 12), (488, 464)]
    full = evolve_rules_batch(E, configs, rules, 9, iso=True, T=10, reduce="none")
    means = evolve_rules_batch(E, configs, rules, 9, iso=True, T=10, reduce="node_mean")
    assert np.array_equal(means, full.astype(np.float32).mean(axis=-1))


@pytest.mark.skipif(not tc.backends.mps.is_available(), reason="MPS not available")
def test_mps_parity(setup):
    E, configs = setup
    rules = get_nonequiv_rules(3)
    cpu = evolve_rules_batch(E, configs, rules, 3, iso=True, T=15, reduce="none", device="cpu")
    mps = evolve_rules_batch(E, configs, rules, 3, iso=True, T=15, reduce="none", device="mps")
    assert np.array_equal(cpu, mps)
