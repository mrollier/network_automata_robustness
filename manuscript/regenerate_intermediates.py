"""Regenerate the manuscript's HDF5 intermediates, seeded and batched.

Replaces the heavy RUN_AGAIN cells of essential_metrics-figures.ipynb.
Group paths and dataset names replicate the legacy notebook cells exactly,
so the figure cells read the files unchanged.

Products (files land in manuscript/data/):
  time-series   state_density_time_series.h5 + defect_density_time_series.h5
  vs-init-dens  final_state_densities.h5 + final_defect_densities.h5
                (group various_init_state_dens; ~110 MB each, kept out of git)
  vs-rewiring   the same two files (group various_rewiring_probs)
  hw-bs         hamming_weights.h5 + boolean_sensitivities.h5

Usage:
  python regenerate_intermediates.py all --seed 42
  python regenerate_intermediates.py time-series hw-bs

Differences from the legacy cells (new random ensembles either way):
  - fully seeded (per-product, per-graph substreams);
  - each state/defect file pair comes from the same trajectories
    (lockstep twins via llna.sweep.evolve_defect_pairs);
  - defect positions are drawn without replacement within a config batch;
  - raw ensemble arrays are stored as float32.
"""

from __future__ import annotations

import argparse
import datetime
import json
import random
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import h5py
import igraph as ig
import numpy as np
import torch as tc

from llna import __version__ as llna_version
from llna.analysis import (
    boolean_sens,
    hamming_weight,
    init_config_with_dens,
    median_and_percentiles_over_ensemble,
)
from llna.networks import create_2d_torus_lattice, watts_strogatz_rewire
from llna.rules import binary_indices, get_nonequiv_rules, return_life_like_dict
from llna.sweep import evolve_defect_pairs

DATA_DIR = Path(__file__).resolve().parent / "data"

# Rule under study in the manuscript's topology-generalisation figures.
MORLEY = (328, 52)
RESOLUTION = 9

# (display name, rewiring probability; None = pristine lattice). The names
# feed the HDF5 group paths, so they must match the notebook verbatim.
MORLEY_TYPES = [("Lattice", None), ("Small World ($p=0.2$)", 0.2), ("Random", 1.0)]
TS_DENS = [(0.25, "1f4"), (0.5, "2f4"), (0.75, "3f4")]

# Product ids keep the RNG substreams of the four products disjoint.
P_TIME_SERIES, P_VS_INIT_DENS, P_VS_REWIRING, P_HW_BS = 1, 2, 3, 4


@dataclass(frozen=True)
class Params:
    L: int = 30
    degree: int = 8
    num_graphs: int = 30
    num_config: int = 30
    T: int = 100
    delta_t: int = 10
    dens_res: int = 51
    rewiring_res: int = 51

    @property
    def num_nodes(self) -> int:
        return self.L**2


QUICK = Params(L=10, num_graphs=3, num_config=4, T=20, delta_t=5, dens_res=7, rewiring_res=5)


def rng_for(*key: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(list(key)))


def pyrandom_for(*key: int) -> random.Random:
    return random.Random(int(np.random.SeedSequence(list(key)).generate_state(1)[0]))


def edge_tensor(g: ig.Graph) -> tc.Tensor:
    gd = g.copy()
    gd.to_directed()
    return tc.tensor(gd.get_edgelist(), dtype=tc.long).T


def morley_edge_tensors(p: Params, rewiring_prob: float | None, seed: int, product: int, type_idx: int):
    """One directed edge tensor per graph in the ensemble (lattice: shared)."""
    lattice = create_2d_torus_lattice(p.L, degree=p.degree)
    if rewiring_prob is None:
        return [edge_tensor(lattice)] * p.num_graphs
    return [
        edge_tensor(
            watts_strogatz_rewire(lattice, rewiring_prob, rng=pyrandom_for(seed, product, type_idx, g))
        )
        for g in range(p.num_graphs)
    ]


def config_defect_block(rng: np.random.Generator, dens_values, M: int, N: int):
    """[D*M, N] clean configs (dens-major) and single-defect twins."""
    clean = np.stack([init_config_with_dens(N, d, rng=rng) for d in dens_values for _ in range(M)])
    defected = clean.copy()
    for b in range(len(dens_values)):
        rows = np.arange(b * M, (b + 1) * M)
        positions = rng.choice(N, size=M, replace=False)
        defected[rows, positions] ^= 1
    return clean, defected


def evolve_type(edges_per_graph, dens_values, p: Params, device: str, seed: int, product: int, type_idx: int):
    """Evolve every (graph, density, config) pair of one graph type.

    Returns (state, defect) node-mean arrays [D, G*M, T+1] float32 with the
    ensemble axis ordered (graph, config) like the legacy vstack loops.
    """
    D, M = len(dens_values), p.num_config
    state = np.empty((D, p.num_graphs * M, p.T + 1), dtype=np.float32)
    defect = np.empty_like(state)
    for g, E in enumerate(edges_per_graph):
        rng = rng_for(seed, product, type_idx, g, 1)
        clean, defected = config_defect_block(rng, dens_values, M, p.num_nodes)
        s, d = evolve_defect_pairs(E, clean, defected, [MORLEY], RESOLUTION, iso=True, T=p.T, device=device)
        state[:, g * M : (g + 1) * M] = s[0].reshape(D, M, p.T + 1)
        defect[:, g * M : (g + 1) * M] = d[0].reshape(D, M, p.T + 1)
    return state, defect


def provenance(seed: int, p: Params) -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        commit = "unknown"
    return {
        "generator": "manuscript/regenerate_intermediates.py",
        "seed": seed,
        "params": json.dumps(vars(p) | {"num_nodes": p.num_nodes}),
        "llna_version": llna_version,
        "git_commit": commit,
        "created_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def write_group(path: Path, group_path: str, datasets: dict, attrs: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "a") as f:
        if group_path in f:
            del f[group_path]
        group = f.create_group(group_path)
        for name, data in datasets.items():
            group.create_dataset(name, data=data)
        group.attrs.update(attrs)


def morley_group_root(p: Params) -> str:
    beta, sigma = MORLEY
    return f"/r{RESOLUTION}beta{beta}sigma{sigma}/N{p.num_nodes}T{p.T}M{p.num_config}num_graphs{p.num_graphs}"


def make_time_series(
    p: Params | None = None, seed: int = 42, device: str = "cpu", outdir: Path = DATA_DIR
) -> None:
    p = p or Params()
    dens_values = [d for d, _ in TS_DENS]
    attrs = provenance(seed, p)
    for type_idx, (name, prob) in enumerate(MORLEY_TYPES):
        print(f"[time-series] {name}")
        edges = morley_edge_tensors(p, prob, seed, P_TIME_SERIES, type_idx)
        state, defect = evolve_type(edges, dens_values, p, device, seed, P_TIME_SERIES, type_idx)
        for fname, raw in [
            ("state_density_time_series.h5", state),
            ("defect_density_time_series.h5", defect),
        ]:
            for b, (_, label) in enumerate(TS_DENS):
                q1, q2, q3 = np.quantile(raw[b].astype(np.float64), [0.25, 0.5, 0.75], axis=0)
                group = f"{morley_group_root(p)}/{name.replace(' ', '_')}/init_dens_{label}"
                write_group(
                    outdir / fname,
                    group,
                    {"q1_of_means": q1, "q2_of_means": q2, "q3_of_means": q3},
                    attrs,
                )


def make_vs_init_dens(
    p: Params | None = None, seed: int = 42, device: str = "cpu", outdir: Path = DATA_DIR
) -> None:
    p = p or Params()
    dens_values = np.linspace(0, 1, p.dens_res)
    attrs = provenance(seed, p)
    raws, medians, lowers, uppers = [], [], [], []
    for type_idx, (name, prob) in enumerate(MORLEY_TYPES):
        print(f"[vs-init-dens] {name}")
        edges = morley_edge_tensors(p, prob, seed, P_VS_INIT_DENS, type_idx)
        state, defect = evolve_type(edges, dens_values, p, device, seed, P_VS_INIT_DENS, type_idx)
        raws.append((state, defect))
        stats = [
            [median_and_percentiles_over_ensemble(raw[b], p.delta_t, time_axis=1) for b in range(p.dens_res)]
            for raw in (state, defect)
        ]
        medians.append([[s[0] for s in per_raw] for per_raw in stats])
        lowers.append([[s[1] for s in per_raw] for per_raw in stats])
        uppers.append([[s[2] for s in per_raw] for per_raw in stats])
    files = [("final_state_densities.h5", "dens", 0), ("final_defect_densities.h5", "damage", 1)]
    for fname, prefix, which in files:
        write_group(
            outdir / fname,
            f"various_init_state_dens{morley_group_root(p)}",
            {
                "init_dens_array": dens_values,
                f"{prefix}_list_per_init_dens_per_graph": np.stack([r[which] for r in raws]),
                "median_list_per_graph": np.array([m[which] for m in medians], dtype=np.float64),
                "lw_perc_list_per_graph": np.array([lo[which] for lo in lowers], dtype=np.float64),
                "up_perc_list_per_graph": np.array([up[which] for up in uppers], dtype=np.float64),
            },
            attrs,
        )


def make_vs_rewiring(
    p: Params | None = None, seed: int = 42, device: str = "cpu", outdir: Path = DATA_DIR
) -> None:
    p = p or Params()
    probs = np.logspace(-2, 0, p.rewiring_res)
    dens_values = [d for d, _ in TS_DENS]
    attrs = provenance(seed, p)
    lattice = create_2d_torus_lattice(p.L, degree=p.degree)
    D, M = len(dens_values), p.num_config
    # [D, P, ens, T+1]; the same graphs serve all init densities (legacy behaviour)
    state = np.empty((D, p.rewiring_res, p.num_graphs * M, p.T + 1), dtype=np.float32)
    defect = np.empty_like(state)
    for i, prob in enumerate(probs):
        print(f"[vs-rewiring] p = {prob:.3f} ({i + 1}/{p.rewiring_res})")
        for g in range(p.num_graphs):
            ws = watts_strogatz_rewire(lattice, prob, rng=pyrandom_for(seed, P_VS_REWIRING, i, g))
            rng = rng_for(seed, P_VS_REWIRING, i, g, 1)
            clean, defected = config_defect_block(rng, dens_values, M, p.num_nodes)
            s, d = evolve_defect_pairs(
                edge_tensor(ws), clean, defected, [MORLEY], RESOLUTION, iso=True, T=p.T, device=device
            )
            state[:, i, g * M : (g + 1) * M] = s[0].reshape(D, M, p.T + 1)
            defect[:, i, g * M : (g + 1) * M] = d[0].reshape(D, M, p.T + 1)
    files = [("final_state_densities.h5", "dens", state), ("final_defect_densities.h5", "damage", defect)]
    for fname, prefix, raw in files:
        stats = np.array(
            [
                [
                    median_and_percentiles_over_ensemble(raw[b, i], p.delta_t, time_axis=1)
                    for i in range(p.rewiring_res)
                ]
                for b in range(D)
            ],
            dtype=np.float64,
        )  # [D, P, 3]
        write_group(
            outdir / fname,
            f"various_rewiring_probs{morley_group_root(p)}",
            {
                "rewiring_prob_array": probs,
                "init_dens_list": np.array(dens_values),
                # legacy stored these time-major: [D, P, T+1, ensemble]
                f"{prefix}_list_per_rewiring_prob_per_init_dens": raw.transpose(0, 1, 3, 2),
                "median_list_per_init_dens": stats[:, :, 0],
                "lw_perc_list_per_init_dens": stats[:, :, 1],
                "up_perc_list_per_init_dens": stats[:, :, 2],
            },
            attrs,
        )


def make_hw_bs(p: Params | None = None, seed: int = 42, outdir: Path = DATA_DIR, resolution: int = 5) -> None:
    p = p or Params()
    """Degree-distribution-weighted rule metrics over three graph ensembles."""
    lattice = create_2d_torus_lattice(p.L, degree=p.degree)
    num_edges = p.num_nodes * p.degree // 2

    def er_graph(g: int) -> ig.Graph:
        random.seed(int(np.random.SeedSequence([seed, P_HW_BS, 2, g]).generate_state(1)[0]))
        while True:
            candidate = ig.Graph.Erdos_Renyi(n=p.num_nodes, m=num_edges)
            if candidate.is_connected():
                return candidate

    ensembles = [
        ("Toroidal Lattice", [lattice] * p.num_graphs),
        (
            "Small World",
            [
                watts_strogatz_rewire(lattice, 0.2, rng=pyrandom_for(seed, P_HW_BS, 1, g))
                for g in range(p.num_graphs)
            ],
        ),
        ("Random", [er_graph(g) for g in range(p.num_graphs)]),
    ]

    rules = get_nonequiv_rules(resolution)
    all_degree_dists = {
        name: [np.bincount(g.degree())[1:] / p.num_nodes for g in graphs] for name, graphs in ensembles
    }
    max_degree = max(len(w) for dists in all_degree_dists.values() for w in dists)
    needed = sorted({k + 1 for dists in all_degree_dists.values() for w in dists for k in np.nonzero(w)[0]})
    print(f"[hw-bs] evaluating {len(rules)} rules x {len(needed)} degrees")
    hw_table = np.zeros((len(rules), max_degree + 1))
    bs_table = np.zeros((len(rules), max_degree + 1))
    for r, (beta, sigma) in enumerate(rules):
        b_set, s_set = binary_indices(int(beta)), binary_indices(int(sigma))
        for k in needed:
            hw_table[r, k] = hamming_weight(resolution, b_set, s_set, k, norm=True, iso=True)
            bs_table[r, k] = boolean_sens(resolution, b_set, s_set, k, norm_degree=False, iso=True)

    attrs = provenance(seed, p) | {"resolution": resolution}
    for name, _ in ensembles:
        # per graph: metrics weighted by that graph's degree distribution -> [G, rules]
        weights = np.zeros((len(all_degree_dists[name]), max_degree + 1))
        for g, w in enumerate(all_degree_dists[name]):
            weights[g, 1 : len(w) + 1] = w
        hw_per_graph = weights @ hw_table.T
        bs_per_graph = weights @ bs_table.T
        group = f"/{name}/N{p.num_nodes}M{p.num_config}num_graphs{p.num_graphs}"
        for path, table, prefix in [
            (outdir / "hamming_weights.h5", hw_per_graph, "hw"),
            (outdir / "boolean_sensitivities.h5", bs_per_graph, "bs"),
        ]:
            write_group(
                path,
                group,
                {
                    f"{prefix}_medians": np.median(table, axis=0),
                    f"{prefix}_q1": np.quantile(table, 0.25, axis=0),
                    f"{prefix}_q3": np.quantile(table, 0.75, axis=0),
                },
                attrs,
            )


PRODUCTS = ["time-series", "vs-init-dens", "vs-rewiring", "hw-bs"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("products", nargs="+", choices=[*PRODUCTS, "all"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--quick", action="store_true", help="tiny parameters, for smoke tests")
    parser.add_argument("--outdir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    assert tuple(return_life_like_dict()["morley"]) == MORLEY
    p = QUICK if args.quick else Params()
    products = PRODUCTS if "all" in args.products else args.products
    for product in products:
        t0 = time.perf_counter()
        if product == "time-series":
            make_time_series(p, args.seed, args.device, args.outdir)
        elif product == "vs-init-dens":
            make_vs_init_dens(p, args.seed, args.device, args.outdir)
        elif product == "vs-rewiring":
            make_vs_rewiring(p, args.seed, args.device, args.outdir)
        elif product == "hw-bs":
            make_hw_bs(p, args.seed, args.outdir)
        print(f"[{product}] done in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
