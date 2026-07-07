"""Capture characterization fixtures from the CURRENT (pre-refactor) implementation.

Run once from the repo root, before any behavioural change to src/:

    /opt/miniconda3/envs/impact-analysis/bin/python tests/characterization/capture_fixtures.py

Writes .npz files into tests/characterization/fixtures/. Each fixture stores the
*inputs* (edge tensors, configs, rule specs) together with the outputs, so the
pytest suite can replay the stored inputs through refactored code and demand
exact equality. Trajectories and one-hot encodings only ever contain exact 0/1
values, so they are stored as uint8 without loss.

Deterministic: fixed seeds, and graphs are stored as edge lists so fixture
validity does not depend on graph-generator determinism.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import igraph as ig
import numpy as np
import torch as tc

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = Path(os.environ.get("FIXTURE_DIR", Path(__file__).resolve().parent / "fixtures"))
FIXTURE_DIR.mkdir(exist_ok=True, parents=True)

from llna.analysis import (  # noqa: E402
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
from llna.automata import LLNA  # noqa: E402
from llna.networks import create_2d_torus_lattice, watts_strogatz_rewire  # noqa: E402
from llna.rules import (  # noqa: E402
    binary_indices,
    bw_symmetric_eca,
    eca_is_llna,
    eca_to_llna,
    get_nonequiv_rules,
    lr_symmetric_eca,
    return_equivalent_rule,
)

T_STEPS = 12

# (name, resolution, B_set, S_set, iso)
RULE_BATTERY = [
    ("R2", 2, [1], [0, 1], False),
    ("R2_emptyS", 2, [0], [], False),
    ("R3", 3, [1], [1, 2], True),
    ("R5", 5, [1, 2, 3], [0, 4], True),
    ("R9_anneal", 9, binary_indices(488), binary_indices(464), True),
    ("R9_noniso", 9, binary_indices(488), binary_indices(464), False),
    ("R9_life", 9, binary_indices(8), binary_indices(12), True),
]


def edge_tensor(graph: ig.Graph) -> tc.Tensor:
    """Directed edge tensor [2, M] with both directions, as LLNA expects."""
    g = graph.copy()
    g.to_directed()
    return tc.tensor(g.get_edgelist(), dtype=tc.long).T


def build_graphs() -> dict:
    import random

    graphs = {}
    graphs["torus8"] = create_2d_torus_lattice(6, degree=8)  # N=36
    graphs["torus4"] = create_2d_torus_lattice(6, degree=4)  # N=36
    random.seed(7)
    graphs["ws"] = watts_strogatz_rewire(graphs["torus8"], 0.15)
    random.seed(11)
    while True:
        er = ig.Graph.Erdos_Renyi(n=40, p=0.15)
        if er.is_connected():
            break
    graphs["er"] = er
    return graphs


def make_configs(n_nodes: int, n_random: int = 4) -> np.ndarray:
    """Random configs at density 0.5 plus all-dead, all-alive, single-alive."""
    rand = (np.random.rand(n_random, n_nodes) < 0.5).astype(np.int64)
    special = np.zeros((3, n_nodes), dtype=np.int64)
    special[1] = 1
    special[2, 0] = 1
    return np.concatenate([rand, special], axis=0)


def check_binary(arr: np.ndarray, label: str) -> np.ndarray:
    assert np.isin(arr, [0.0, 1.0]).all(), f"{label} contains non-binary values"
    return arr.astype(np.uint8)


def capture_trajectories(graphs: dict) -> None:
    np.random.seed(2026)
    out = {}
    case_graph = {
        "R2": "torus8",
        "R2_emptyS": "torus8",
        "R3": "torus4",
        "R5": "ws",
        "R9_anneal": "torus8",
        "R9_noniso": "er",
        "R9_life": "er",
    }
    for name, res, b_set, s_set, iso in RULE_BATTERY:
        graph = graphs[case_graph[name]]
        E = edge_tensor(graph)
        configs = make_configs(graph.vcount())
        model = LLNA(res, b_set, s_set, iso=iso)
        traj_int = model.forward(E, tc.tensor(configs), T=T_STEPS).numpy()
        traj_f32 = model.forward(E, tc.tensor(configs.astype(np.float32)), T=T_STEPS).numpy()
        out[f"{name}__E"] = E.numpy()
        out[f"{name}__configs"] = configs.astype(np.uint8)
        out[f"{name}__traj_int"] = check_binary(traj_int, name)
        out[f"{name}__traj_f32"] = check_binary(traj_f32, name)
        print(f"  trajectories {name}: graph={case_graph[name]} N={graph.vcount()} L={len(configs)}")
    np.savez_compressed(FIXTURE_DIR / "trajectories.npz", **out)


def capture_star_boundaries() -> None:
    """Star graphs whose hub degree equals the resolution: hub densities hit the
    interval boundaries k/R exactly, the risky spots for the encoding."""
    out = {}
    star_battery = [
        ("R2", 2, [1], [0, 1], False),
        ("R3", 3, [1], [1, 2], True),
        ("R5", 5, [1, 2, 3], [0, 4], True),
        ("R9_anneal", 9, binary_indices(488), binary_indices(464), True),
    ]
    for name, res, b_set, s_set, iso in star_battery:
        n = res + 1
        star = ig.Graph.Star(n, mode="undirected", center=0)
        E = edge_tensor(star)
        n_all = 2**n
        configs = ((np.arange(n_all)[:, None] >> np.arange(n)) & 1).astype(np.int64)
        model = LLNA(res, b_set, s_set, iso=iso)
        traj = model.forward(E, tc.tensor(configs), T=2).numpy()
        out[f"{name}__E"] = E.numpy()
        out[f"{name}__configs"] = configs.astype(np.uint8)
        out[f"{name}__traj"] = check_binary(traj, f"star {name}")
        print(f"  star {name}: N={n} L={n_all}")
    np.savez_compressed(FIXTURE_DIR / "star_boundaries.npz", **out)


def capture_interval_encoding() -> None:
    """Torch (float32 input, as produced by the conv) and numpy (float64 input)
    one-hot interval encodings on a grid including exact boundaries k/R and
    their nearest float neighbours."""
    out = {}
    for res, iso in [(2, False), (3, False), (3, True), (5, True), (9, False), (9, True)]:
        bounds = np.arange(res + 1) / res
        grid = np.concatenate(
            [
                np.linspace(0, 1, 257),
                bounds,
                np.nextafter(bounds, -1.0),
                np.nextafter(bounds, 2.0),
            ]
        )
        grid = np.unique(np.clip(grid, 0.0, 1.0))
        model = LLNA(res, [0], [0], iso=iso)
        torch_onehot = model.interval_encoding(tc.tensor(grid, dtype=tc.float32).unsqueeze(0)).numpy()[0]
        numpy_onehot = _interval_encoding(res, grid[np.newaxis, :].astype(np.float64), iso=iso)[0]
        key = f"R{res}_{'iso' if iso else 'noniso'}"
        out[f"{key}__grid"] = grid
        out[f"{key}__torch_f32"] = check_binary(torch_onehot, key)
        out[f"{key}__numpy_f64"] = numpy_onehot.astype(np.uint8)
        n_diff = int((torch_onehot.astype(np.uint8) != numpy_onehot.astype(np.uint8)).any(axis=1).sum())
        print(f"  interval encoding {key}: grid={len(grid)} torch-vs-numpy row diffs={n_diff}")
    np.savez_compressed(FIXTURE_DIR / "interval_encoding.npz", **out)


def capture_jacobian_lyapunov(graphs: dict) -> None:
    np.random.seed(31)
    out = {}
    for name, graph_name, res, b_set, s_set, iso in [
        ("R5_torus8", "torus8", 5, [1, 2, 3], [0, 4], True),
        ("R9_er", "er", 9, binary_indices(488), binary_indices(464), True),
    ]:
        graph = graphs[graph_name]
        model = LLNA(res, b_set, s_set, iso=iso)
        states = (np.random.rand(graph.vcount()) < 0.5).astype(np.int64)
        J, states_next = jacobian(graph, model, states, return_next=True)
        out[f"{name}__E"] = edge_tensor(graph).numpy()
        out[f"{name}__states"] = states.astype(np.uint8)
        out[f"{name}__J"] = J.astype(np.uint8)
        out[f"{name}__states_next"] = states_next.astype(np.uint8)
        print(f"  jacobian {name}: N={graph.vcount()}")

    # tangent-space propagation + numerical spectrum (float64 contract)
    graph = graphs["torus8"]
    model = LLNA(5, [1, 2, 3], [0, 4], iso=True)
    states = (np.random.rand(graph.vcount()) < 0.5).astype(np.int64)
    Yt = calculate_Yt(graph, model, states, T=8)
    with np.errstate(divide="ignore"):
        lambdas = lyapunov_spectrum(Yt, T=8)
    out["Yt__states"] = states.astype(np.uint8)
    out["Yt__value"] = Yt
    out["Yt__lambdas"] = lambdas
    print(
        f"  calculate_Yt: N={graph.vcount()} T=8, spectrum range "
        f"[{np.min(lambdas[np.isfinite(lambdas)]):.3f}, {np.max(lambdas):.3f}]"
    )

    # analytical constant-J ECA spectra
    for rule in [150, 90, 105, 60]:
        out[f"analytical_rule{rule}_N64"] = lyapunov_spectrum_analytical(rule, 64)
    vals, pct = lyapunov_spectrum_analytical(150, 63, return_finite_pct=True)
    out["analytical_rule150_N63"] = vals
    out["analytical_rule150_N63_pct"] = np.array([pct])
    np.savez_compressed(FIXTURE_DIR / "jacobian_lyapunov.npz", **out)


def capture_rule_metrics() -> None:
    """Both duplicate implementations of Hamming weight / Boolean sensitivity,
    plus mean-field and analytical-Derrida curves."""
    out = {}
    rows = []
    degrees = [3, 4, 8]
    for _name, res, b_set, s_set, iso in RULE_BATTERY:
        model = LLNA(res, b_set, s_set, iso=iso)
        for degree in degrees:
            hw_m = model.hamming_weight(degree, norm=True)
            hw_f = hamming_weight(res, b_set, s_set, degree, norm=True, iso=iso)
            bs_m = model.boolean_sens(degree, norm_degree=True)
            bs_f_norm = boolean_sens(res, b_set, s_set, degree, norm_degree=True, current_dens=0.5, iso=iso)
            bs_f_raw = boolean_sens(res, b_set, s_set, degree, norm_degree=False, current_dens=0.5, iso=iso)
            rows.append([hw_m, hw_f, bs_m, bs_f_norm, bs_f_raw])
    out["metric_names"] = np.array(["hw_method", "hw_func", "bs_method", "bs_func_norm", "bs_func_raw"])
    out["case_names"] = np.array([f"{name}_d{d}" for name, *_ in RULE_BATTERY for d in degrees])
    out["values"] = np.array(rows, dtype=np.float64)

    dens_grid = np.linspace(0, 1, 101)
    out["meanfield__dens_grid"] = dens_grid
    out["meanfield__R5_d8"] = mean_field_dens_propagation(5, [1, 2, 3], [0, 4], dens_grid, 8, iso=True)
    out["meanfield__R9_anneal_d8"] = mean_field_dens_propagation(
        9, binary_indices(488), binary_indices(464), dens_grid, 8, iso=True
    )

    delta_grid = np.linspace(0, 1, 9)
    out["derrida__delta_grid"] = delta_grid
    out["derrida__R3_d4"] = np.array(
        [derrida_map_analytical(3, [1], [1, 2], d, 4, iso=True) for d in delta_grid]
    ).squeeze()
    print(f"  rule metrics: {len(rows)} rows; mean-field + analytical Derrida curves")
    np.savez_compressed(FIXTURE_DIR / "rule_metrics.npz", **out)


def capture_rule_enumeration() -> None:
    out = {}
    for res in [2, 3, 5]:
        nonequiv, self_equiv = get_nonequiv_rules(res, return_self_equiv=True)
        out[f"nonequiv_R{res}"] = np.array(nonequiv, dtype=np.int64)
        out[f"self_equiv_R{res}"] = np.array(self_equiv, dtype=np.int64)
        print(f"  enumeration R={res}: {len(nonequiv)} non-equivalent, {len(self_equiv)} self-equivalent")

    samples = []
    for res, b_set, s_set in [
        (2, [1], [0, 1]),
        (3, [1], [1, 2]),
        (5, [1, 2, 3], [0, 4]),
        (9, binary_indices(488), binary_indices(464)),
        (9, [], []),
        (9, list(range(9)), list(range(9))),
    ]:
        beta = sum(2**i for i in b_set)
        sigma = sum(2**i for i in s_set)
        be, se = return_equivalent_rule(res, b_set, s_set, return_decimals=True)
        samples.append([res, beta, sigma, be, se])
    out["equivalent_rule_samples"] = np.array(samples, dtype=np.int64)

    ecas = np.arange(256)
    out["eca_lr"] = np.array([lr_symmetric_eca(int(e)) for e in ecas], dtype=np.int64)
    out["eca_bw"] = np.array([bw_symmetric_eca(int(e)) for e in ecas], dtype=np.int64)
    out["eca_is_llna"] = np.array([eca_is_llna(int(e)) for e in ecas], dtype=np.uint8)
    llna_rows = []
    for e in ecas:
        if eca_is_llna(int(e)):
            b_set, s_set = eca_to_llna(int(e))
            llna_rows.append([int(e), sum(2**i for i in b_set), sum(2**i for i in s_set)])
    out["eca_to_llna"] = np.array(llna_rows, dtype=np.int64)
    print(f"  ECA helpers: {len(llna_rows)} ECAs map to LLNA")
    np.savez_compressed(FIXTURE_DIR / "rules_enum.npz", **out)


def capture_rng_contract() -> None:
    np.random.seed(1234)
    a = init_config_with_dens(100, 0.37)
    b = init_config_with_dens(64, 0.5)
    np.savez_compressed(
        FIXTURE_DIR / "rng_contract.npz",
        init_dens_037_N100=a.astype(np.uint8),
        init_dens_05_N64=b.astype(np.uint8),
    )
    print(f"  RNG contract: densities {a.mean():.2f}, {b.mean():.2f}")


def write_manifest() -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT
    ).stdout.strip()
    manifest = {
        "captured_at_commit": commit,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": tc.__version__,
        "torch_geometric": __import__("torch_geometric").__version__,
        "igraph": ig.__version__,
        "note": "Fixtures pin pre-refactor behaviour; exact-equality contract.",
    }
    (FIXTURE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"  manifest written (commit {commit[:8]})")


if __name__ == "__main__":
    tc.manual_seed(0)
    print("Capturing characterization fixtures...")
    graphs = build_graphs()
    capture_trajectories(graphs)
    capture_star_boundaries()
    capture_interval_encoding()
    capture_jacobian_lyapunov(graphs)
    capture_rule_metrics()
    capture_rule_enumeration()
    capture_rng_contract()
    write_manifest()
    total = sum(f.stat().st_size for f in FIXTURE_DIR.glob("*"))
    print(f"Done. Fixture dir size: {total / 1024:.0f} KiB")
