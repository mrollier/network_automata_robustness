"""Replay stored inputs through the current code and demand exact equality
with the pre-refactor fixtures (captured at Phase 0). These tests are the
contract for the engine swap, dedupe, and analysis split.

Everything integer-valued or produced by the torch engine is compared
bit-exactly — that holds across platforms. Float64 outputs of libm/LAPACK
(metrics, mean-field curves, Lyapunov spectra) differ by 1-2 ULP between
platforms, so those few assertions use numerical-precision tolerances."""

import igraph as ig
import numpy as np
import pytest
import torch as tc
from capture_fixtures import RULE_BATTERY

from llna.analysis import (
    _interval_encoding,
    boolean_sens,
    calculate_Yt,
    derrida_map_analytical,
    hamming_weight,
    init_config_with_dens,
    jacobian,
    lyapunov_spectrum,
    lyapunov_spectrum_analytical,
    mean_field_dens_propagation,
)
from llna.automata import LLNA
from llna.rules import (
    binary_indices,
    bw_symmetric_eca,
    eca_is_llna,
    eca_to_llna,
    get_nonequiv_rules,
    lr_symmetric_eca,
    return_equivalent_rule,
)

RULES_BY_NAME = {name: (res, b, s, iso) for name, res, b, s, iso in RULE_BATTERY}


def graph_from_edge_tensor(E: np.ndarray) -> ig.Graph:
    """Rebuild the undirected igraph from a stored directed edge tensor."""
    n = int(E.max()) + 1
    und = {(min(a, b), max(a, b)) for a, b in E.T.tolist()}
    return ig.Graph(n=n, edges=sorted(und))


@pytest.mark.parametrize("name", [c[0] for c in RULE_BATTERY])
def test_forward_trajectories(fx, name):
    d = fx["trajectories"]
    res, b_set, s_set, iso = RULES_BY_NAME[name]
    E = tc.tensor(d[f"{name}__E"])
    configs = d[f"{name}__configs"].astype(np.int64)
    model = LLNA(res, b_set, s_set, iso=iso)
    traj_int = model.forward(E, tc.tensor(configs), T=12).numpy()
    traj_f32 = model.forward(E, tc.tensor(configs.astype(np.float32)), T=12).numpy()
    assert np.array_equal(traj_int, d[f"{name}__traj_int"])
    assert np.array_equal(traj_f32, d[f"{name}__traj_f32"])


@pytest.mark.parametrize("name", ["R2", "R3", "R5", "R9_anneal"])
def test_star_boundary_trajectories(fx, name):
    d = fx["star_boundaries"]
    res, b_set, s_set, iso = RULES_BY_NAME[name]
    E = tc.tensor(d[f"{name}__E"])
    configs = d[f"{name}__configs"].astype(np.int64)
    model = LLNA(res, b_set, s_set, iso=iso)
    traj = model.forward(E, tc.tensor(configs), T=2).numpy()
    assert np.array_equal(traj, d[f"{name}__traj"])


@pytest.mark.parametrize("res,iso", [(2, False), (3, False), (3, True), (5, True), (9, False), (9, True)])
def test_interval_encodings(fx, res, iso):
    d = fx["interval_encoding"]
    key = f"R{res}_{'iso' if iso else 'noniso'}"
    grid = d[f"{key}__grid"]
    model = LLNA(res, [0], [0], iso=iso)
    torch_onehot = model.interval_encoding(tc.tensor(grid, dtype=tc.float32).unsqueeze(0)).numpy()[0]
    numpy_onehot = _interval_encoding(res, grid[np.newaxis, :], iso=iso)[0]
    assert np.array_equal(torch_onehot.astype(np.uint8), d[f"{key}__torch_f32"])
    assert np.array_equal(numpy_onehot.astype(np.uint8), d[f"{key}__numpy_f64"])


@pytest.mark.parametrize("name,rule_name", [("R5_torus8", "R5"), ("R9_er", "R9_anneal")])
def test_jacobian(fx, name, rule_name):
    d = fx["jacobian_lyapunov"]
    res, b_set, s_set, iso = RULES_BY_NAME[rule_name]
    graph = graph_from_edge_tensor(d[f"{name}__E"])
    model = LLNA(res, b_set, s_set, iso=iso)
    states = d[f"{name}__states"].astype(np.int64)
    J, states_next = jacobian(graph, model, states, return_next=True)
    assert np.array_equal(J.astype(np.uint8), d[f"{name}__J"])
    assert np.array_equal(states_next.astype(np.uint8), d[f"{name}__states_next"])


def test_calculate_Yt_and_spectrum(fx):
    d = fx["jacobian_lyapunov"]
    graph = graph_from_edge_tensor(d["R5_torus8__E"])
    model = LLNA(5, [1, 2, 3], [0, 4], iso=True)
    states = d["Yt__states"].astype(np.int64)
    Yt = calculate_Yt(graph, model, states, T=8)
    assert np.array_equal(Yt, d["Yt__value"])
    with np.errstate(divide="ignore"):
        lambdas = lyapunov_spectrum(Yt, T=8)
    # Singular values at the SVD noise floor (sigma ~ N * eps * sigma_max)
    # depend on the LAPACK build, so the log-domain spectrum is only
    # reproducible down to the standard backward-error bound. Compare in the
    # sigma domain with that bound as the absolute tolerance.
    sig, sig_fix = np.exp(lambdas * 8), np.exp(d["Yt__lambdas"] * 8)
    np.testing.assert_allclose(
        sig, sig_fix, rtol=1e-9, atol=sig_fix.max() * len(sig_fix) * np.finfo(float).eps
    )


def test_analytical_spectra(fx):
    d = fx["jacobian_lyapunov"]
    for rule in [150, 90, 105, 60]:
        np.testing.assert_allclose(
            lyapunov_spectrum_analytical(rule, 64), d[f"analytical_rule{rule}_N64"], rtol=1e-12, atol=1e-12
        )
    vals, pct = lyapunov_spectrum_analytical(150, 63, return_finite_pct=True)
    np.testing.assert_allclose(vals, d["analytical_rule150_N63"], rtol=1e-12, atol=1e-12)
    assert pct == d["analytical_rule150_N63_pct"][0]


def test_rule_metrics_both_implementations(fx):
    """The numpy analysis functions are the canonical implementation and must
    reproduce their fixture columns exactly. The LLNA methods delegate to them
    since Phase 5, so method == function; the old torch-encoding method values
    (columns hw_method / bs_method) remain in the fixture as a historical
    record of the float32-boundary behaviour but are no longer a contract."""
    d = fx["rule_metrics"]
    expected = d["values"]
    i = 0
    for name, res, b_set, s_set, iso in RULE_BATTERY:
        model = LLNA(res, b_set, s_set, iso=iso)
        for degree in [3, 4, 8]:
            hw_func = hamming_weight(res, b_set, s_set, degree, norm=True, iso=iso)
            bs_func_norm = boolean_sens(
                res, b_set, s_set, degree, norm_degree=True, current_dens=0.5, iso=iso
            )
            bs_func_raw = boolean_sens(
                res, b_set, s_set, degree, norm_degree=False, current_dens=0.5, iso=iso
            )
            tol = dict(rel=1e-12, abs=1e-15)
            assert hw_func == pytest.approx(expected[i, 1], **tol), f"{name} degree {degree}"
            assert bs_func_norm == pytest.approx(expected[i, 3], **tol), f"{name} degree {degree}"
            assert bs_func_raw == pytest.approx(expected[i, 4], **tol), f"{name} degree {degree}"
            assert model.hamming_weight(degree, norm=True) == hw_func
            assert model.boolean_sens(degree, norm_degree=True) == bs_func_norm
            i += 1


def test_meanfield_and_derrida_curves(fx):
    d = fx["rule_metrics"]
    dens_grid = d["meanfield__dens_grid"]
    np.testing.assert_allclose(
        mean_field_dens_propagation(5, [1, 2, 3], [0, 4], dens_grid, 8, iso=True),
        d["meanfield__R5_d8"],
        rtol=1e-12,
        atol=1e-15,
    )
    np.testing.assert_allclose(
        mean_field_dens_propagation(9, binary_indices(488), binary_indices(464), dens_grid, 8, iso=True),
        d["meanfield__R9_anneal_d8"],
        rtol=1e-12,
        atol=1e-15,
    )
    delta_grid = d["derrida__delta_grid"]
    derrida = np.array([derrida_map_analytical(3, [1], [1, 2], x, 4, iso=True) for x in delta_grid]).squeeze()
    np.testing.assert_allclose(derrida, d["derrida__R3_d4"], rtol=1e-12, atol=1e-15)


def test_rule_enumeration(fx):
    d = fx["rules_enum"]
    for res in [2, 3, 5]:
        nonequiv, self_equiv = get_nonequiv_rules(res, return_self_equiv=True)
        assert np.array_equal(np.array(nonequiv, dtype=np.int64), d[f"nonequiv_R{res}"])
        assert np.array_equal(np.array(self_equiv, dtype=np.int64), d[f"self_equiv_R{res}"])


def test_equivalent_rule_samples(fx):
    for res, beta, sigma, be_expected, se_expected in fx["rules_enum"]["equivalent_rule_samples"]:
        be, se = return_equivalent_rule(
            int(res), binary_indices(int(beta)), binary_indices(int(sigma)), return_decimals=True
        )
        assert (be, se) == (be_expected, se_expected)


def test_eca_helpers(fx):
    d = fx["rules_enum"]
    ecas = np.arange(256)
    assert np.array_equal(np.array([lr_symmetric_eca(int(e)) for e in ecas]), d["eca_lr"])
    assert np.array_equal(np.array([bw_symmetric_eca(int(e)) for e in ecas]), d["eca_bw"])
    assert np.array_equal(np.array([eca_is_llna(int(e)) for e in ecas]).astype(np.uint8), d["eca_is_llna"])
    for eca, beta, sigma in d["eca_to_llna"]:
        b_set, s_set = eca_to_llna(int(eca))
        assert sum(2**i for i in b_set) == beta and sum(2**i for i in s_set) == sigma


def test_init_config_rng_contract(fx):
    d = fx["rng_contract"]
    np.random.seed(1234)
    a = init_config_with_dens(100, 0.37)
    b = init_config_with_dens(64, 0.5)
    assert np.array_equal(a.astype(np.uint8), d["init_dens_037_N100"])
    assert np.array_equal(b.astype(np.uint8), d["init_dens_05_N64"])
