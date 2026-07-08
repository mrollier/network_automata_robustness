# Performance benchmarks — 2026-07 cleanup

Machine: Apple Silicon (macOS), conda env `impact-analysis`, torch 2.7 CPU +
MPS. All equivalences below are *exact* (bit-for-bit values), enforced by the
test suite (`tests/test_characterization.py`, `tests/test_sweep.py`).

## Rule enumeration (`llna.rules.get_nonequiv_rules`)

| Resolution | Rules | Old (loop + list scans) | New (vectorised bit tables) |
|---|---|---|---|
| 5 | 528 | ~1 s | < 1 ms |
| 9 | 131,328 | **hours** (the committed table was precomputed for this reason) | **36 ms** |

Output verified equal to the committed `data/rule_tables/all_nonequiv_res9_rules.npy`
including row order.

## Single-model dynamics (`LLNA.forward`, N=900, L=60 configs, T=100)

| Implementation | Time |
|---|---|
| torch_geometric `SimpleConv` (pre-cleanup) | 752 ms |
| `llna.engine.neighbor_density` (`index_add_`) | 787 ms |

Neutral speed; the swap's value is dropping the torch_geometric dependency,
enabling MPS, and unlocking the batched path below. States are exact 0/1, so
neighbour sums are small integers — exact in float32 — which makes
sum-then-divide bit-identical to the message-passing mean.

## Batched rule sweep (528 non-equivalent R=5 rules, WS graph N=900, L=60, T=100)

| Method | Wall time |
|---|---|
| Sequential per-rule loop (historical workflow) | ~145 s (extrapolated from 10 rules) |
| `llna.sweep.evolve_rules_batch`, CPU | **35 s** |
| `llna.sweep.evolve_rules_batch`, MPS | 42 s |

- ~4x from eliminating per-rule Python/kernel overhead; the remaining cost is
  the O(K·L·M) neighbour gather, which is intrinsic to the workload.
- MPS is exact-equal to CPU but slower at this size (scatter-bound); CPU is
  the default device.
- A sparse-CSR matmul density variant was measured at 2.7x slower than the
  gather and rejected.

## End-to-end sweep (`llna sweep`)

The historical `scripts/state_vs-defect-density.py` looped rules x graphs
sequentially (~0.28 s per rule-graph at the scale above -> ~3.5 h for
528 rules x 30 graphs) and could not be interrupted. `llna sweep` runs the
same workload batched (~35 s per graph-ensemble chunk pass, ~18 min total at
default settings), is seeded end-to-end, and resumes from the last completed
rule chunk after interruption.

## Manuscript intermediates (`manuscript/regenerate_intermediates.py`, 2026-07)

The morley-rule RUN_AGAIN cells of `essential_metrics-figures.ipynb` were
rewritten onto `llna.sweep.evolve_defect_pairs` (all densities x configs of a
graph evolve in one batched call; clean/defected twins in lockstep, so each
state h5 and its defect twin come from the same run). Wall times for the full
seed-42 regeneration on CPU:

| Product | Workload | Wall time |
|---|---|---|
| `time-series` | 3 types x 30 graphs x (3 dens x 30 configs) pairs, T=100, N=900 | ~1 min |
| `vs-init-dens` | 3 types x 30 graphs x (51 dens x 30 configs) pairs | ~5 min (3.3 s/graph) |
| `vs-rewiring` | 51 p-values x 30 graphs x (3 dens x 30 configs) pairs | 620 s |
| `hw-bs` | 528 rules x 21 distinct degrees, metric table reused across 90 graphs | 5.8 s |

The legacy cells ran the same workloads with per-graph Python loops and
unbatched `LLNA.forward` calls plus O(n^2) `vstack` accumulation ("takes a
long time!" per their own comments — hours in total); the rewrite finishes
everything in ~16 min and is seeded end-to-end.

Reproduce any row with the snippets in this file's git history or via
`llna sweep --help`.
