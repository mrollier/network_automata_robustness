# Legacy TEP pipeline (archived, unmaintained)

This directory preserves the original study pipeline by Lucas Caldeira
(`lcaldeira`), which generated Temporal Evolution Profiles (TEPs) for the
smallest LLNA family on Barabási–Albert graphs. The current research
(`manuscript/`, `experiments/`) does not use it. It is kept for provenance,
**as-is and unfixed**.

## Contents

| File | Role |
|---|---|
| `README-original.md` | The original project README (its "Project structure" section was never finished) |
| `1-create_nets.py` | Step 1 — generate BA graphs + measures. **Broken**: imports from the `network_analysis` submodule (lcaldeira/network_analysis), which was removed from this repo |
| `2-run_automata.py` | Step 2 — evolve every rule of a resolution family over all graphs, store TEPs. Configured interactively via `eval(input())` |
| `node_measures.py` | Step 3 — per-node defect statistics from TEPs → `data/calc/<rule>.csv.xz` |
| `simulation.py` | Tensor helpers (perturbations, reshaping, pickled-tensor I/O) used by steps 2–3 |
| `datasets.py` | `torch` `Dataset` readers for the graph `.txt` files and the `.tar.xz` TEP archives |
| `run_simulations.sh` | Batch driver with a hardcoded Linux path (non-portable) |

## Where the data went

The pipeline's data (`data/graphs/` — 240 custom graph text files,
`data/info/` — pickled pandas metadata, `data/teps/` — 16 `.tar.xz` archives
of pickled bool tensors) was removed from the repository during the 2026-07
cleanup, including from git history.

Full copies live in:

- `~/Backups/llna-worktree-2026-07-07.tgz` (worktree snapshot)
- `~/Backups/llna-mirror-2026-07-07.git` (complete pre-rewrite git mirror)

## Why unfixed

Running step 1 requires the external `network_analysis` library; steps 2–3
predate the argparse-based tooling in the current package. Reviving the
pipeline would mean re-adding the submodule and porting the scripts — decided
against on 2026-07-07 because no active work consumes TEPs.
