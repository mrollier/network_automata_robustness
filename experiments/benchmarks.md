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

Reproduce any row with the snippets in this file's git history or via
`llna sweep --help`.
