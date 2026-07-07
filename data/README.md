# data/

Curated data files consumed or produced by the living notebooks and the CLI.

## rule_tables/

Non-equivalent rule tables, `all_nonequiv_res{R}_rules.npy`, shape `[K, 2]`
(`beta`, `sigma` integer columns), ascending lexicographic order; one
representative per black-white equivalence class, `K = (4^R + 2^R) / 2`.

Regenerate (or extend to other resolutions) with:

```bash
llna rules --resolution 9 --cache-dir data/rule_tables
```

The committed R=9 table is verified by the test suite to equal the current
enumeration exactly.

## results/

State/defect convergence sweeps in HDF5. Layout per file:

```
resolution{R}/{network-type}/rules            [K, 2] int64   (only in regenerated files)
resolution{R}/{network-type}/state/median     [K] float64
resolution{R}/{network-type}/state/q1, q3     [K] float64
resolution{R}/{network-type}/defect/median    [K] float64
resolution{R}/{network-type}/defect/q1, q3    [K] float64
```

The statistics pool the final `delta_t` timesteps of every ensemble member
(graphs × initial configurations).

| File | Provenance |
|---|---|
| `results_{random,small-world,toroidal-lattice}.h5` | **Legacy (2025)**, produced by the historical CLI whose ensemble statistics pooled the wrong axis (last `delta_t` ensemble *members* including transients, instead of last `delta_t` *timesteps*; fixed 2026-07). Kept byte-identical because the manuscript figure currently reads them. |
| `regenerated_2026/results_*.h5` | Regenerated with `llna sweep` (fixed statistics, seed 42, batched engine). Carry full provenance attrs: parameters, seed, git commit, llna version, timestamps. **The manuscript notebook reads these since 2026-07-07** (approved swap); `regenerated_2026/comparison_legacy_vs_2026.png` visualises the per-rule deltas (corr 0.98–0.997; a few long-transient rules shift by up to ~0.47 in defect median — the legacy statistics pooled the wrong axis for exactly such rules). |
| `state-median-vs-defect-median.h5` | Written by `experiments/state-vs-defect-density.ipynb` (resolution 3). |

Regeneration one-liner per network type:

```bash
llna sweep --network-type random --resolution 5 --L 30 --T 100 \
           --num-graphs 30 --num-init-conf 30 --delta-t 10 --seed 42 \
           --output data/results/regenerated_2026/results_random.h5
```

## Related data elsewhere

- `manuscript/data/` — `.npy` inputs for the manuscript figure notebooks
  (totalitarian rule tables, final-state densities, FSSP success rates),
  produced by `RUN_AGAIN`-guarded cells in those notebooks.
- Legacy TEP archives (graphs/info/teps of the original study) were removed
  from the repository in the 2026-07 cleanup; copies live in the external
  backup (see `legacy/README.md`).
