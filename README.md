# Network automata robustness

Robustness of **Life-Like Network Automata (LLNA)**: how one-bit perturbations
propagate through binary, density-based automata running on networks — measured
through defect densities, Boolean Jacobians, Lyapunov spectra, Derrida maps,
and mean-field theory.

An LLNA generalises Conway's Game of Life to arbitrary graphs: a node's next
state depends on its own state and on which of `R` density intervals the alive
fraction of its neighbourhood falls into (born set `B`, survive set `S`).
Start with [notebooks/01_llna_basics.ipynb](notebooks/01_llna_basics.ipynb).

## Install

```bash
conda env create -f environment.yml     # creates env "llna", installs the package editable
conda activate llna
pytest                                  # 100+ tests, ~5 s
```

Or into an existing Python ≥ 3.11 environment: `pip install -e ".[dev]"`.

## Quickstart

```python
import torch as tc
from llna import LLNA
from llna.networks import create_2d_torus_lattice

graph = create_2d_torus_lattice(12, degree=8)
directed = graph.copy(); directed.to_directed()
E = tc.tensor(directed.get_edgelist(), dtype=tc.long).T

gol = LLNA(9, [3], [2, 3], iso=True)      # Game of Life: R9 B{3} S{2,3}
trajectories = gol.forward(E, soup_tensor, T=100)   # [L, T+1, N]
```

Sweep every non-equivalent rule over a graph ensemble (batched, seeded,
resumable — interrupt and re-run the same command to continue):

```bash
llna rules --resolution 9 --cache-dir data/rule_tables
llna sweep --network-type small-world --resolution 5 --L 30 --T 100 \
           --num-graphs 30 --num-init-conf 30 --seed 42 --output results.h5
```

Performance numbers (rule enumeration: hours → 36 ms at R=9; full R=5 sweep:
~3.5 h → ~18 min) are documented in
[experiments/benchmarks.md](experiments/benchmarks.md).

## Repository layout

| Path | Contents |
|---|---|
| `src/llna/` | The package: `automata` (the LLNA model), `engine` (vectorised dynamics core), `rules` (rule algebra & equivalences), `networks` (graph builders), `sweep` (batched rule sweeps), `io` (resumable HDF5 results), `cli`, `visual`, and `analysis/` (jacobians & Lyapunov, Derrida, sensitivity, mean-field, ensemble statistics, ECA references) |
| `notebooks/` | Explanatory notebooks: LLNA basics · rule space & metrics · robustness metrics |
| `tests/` | Pytest suite incl. characterization fixtures that pin the dynamics bit-for-bit |
| `manuscript/` | Figure-generating notebooks + data for the robustness paper |
| `experiments/` | Living research notebooks (outputs stripped) + benchmarks |
| `data/` | Curated inputs/results — see [data/README.md](data/README.md) |
| `csf-submission/` | The CSF Lyapunov paper (its figure pipeline lives in the separate `exact-lyapunov-spectra` repository) |
| `presentation/` | Talk figure generators |
| `archive/` | Stale one-off notebooks kept for reference (outputs stripped) |
| `legacy/` | The original TEP pipeline by Lucas Caldeira, unmaintained — see [legacy/README.md](legacy/README.md) |

## Reproducing the manuscript figures

`manuscript/essential_metrics-figures.ipynb` and
`manuscript/impact_analysis-figures.ipynb` generate the paper figures into
`manuscript/figures/` (git-ignored). Notes:

- Plot styling uses LaTeX (`text.usetex`); if matplotlib complains about
  `type1cm.sty`, run `sudo tlmgr install type1cm cm-super underscore dvipng`.
- Some heavy intermediate datasets are produced by `RUN_AGAIN`-guarded cells
  inside the notebooks and are not committed; `llna sweep` regenerates the
  state-vs-defect results far faster than the original in-notebook loops.

## Testing & conventions

- `pytest` — includes exact-replay characterization tests: any change to the
  dynamics that alters a single bit of a stored trajectory fails the suite.
- `ruff check` / `ruff format` — configured in `pyproject.toml`
  (`legacy/` and `archive/` are excluded on purpose).
- Randomness: library functions accept an explicit `rng`; the CLI is fully
  seeded (`--seed`) and stores parameters, seed, and git commit as HDF5 attrs.

## Credits

Started by Lucas Caldeira (`lcaldeira`) and Michiel Rollier; the original
TEP-based study is preserved under `legacy/`. Current research: Michiel
Rollier, Ghent University (with Jan Baetens).
