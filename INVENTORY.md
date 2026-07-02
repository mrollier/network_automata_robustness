# INVENTORY — Phase 0 audit

Audit of the current repository against the submitted CSF manuscript
*"Exact Lyapunov spectra of affine cellular automata and the parity rule on
networks"* (Rollier & Baetens). Read-only; nothing in the repository has been
modified. UK English throughout.

Ground truth for figures, tables and claims is the submitted source
[`csf-submission/lyapunov-first_submission_CSF.tex`](csf-submission/lyapunov-first_submission_CSF.tex)
(+ its compiled PDF, present, usable for visual comparison).

---

## 1. Executive summary (the findings that shape the plan)

1. **The figure-generating code is not in this repository.** None of the five
   figure file-stems referenced by the manuscript appear in any `.py` or
   `.ipynb` here (verified by grep across the whole tree). The paper's *Data
   availability* section names a **different** repository,
   `mrollier/exact-lyapunov-spectra`; this repository's git remote is
   `mrollier/network_automata_robustness`. **Building `paper-repo/` is therefore
   largely a fresh implementation of the figure scripts**, guided by (a) the
   closed-form formulas in the paper, which are complete and self-contained, and
   (b) reusable maths fragments found in `src/analysis.py` and one notebook.

2. **Two projects are mixed in this repo.** The CSF Lyapunov paper
   (`csf-submission/`) is the target. A *separate* paper on network-automata
   robustness (`manuscript/` notebooks: essential metrics, impact analysis,
   FSSP, LLNA robustness) is **out of scope** and should not be pulled into
   `paper-repo/`.

3. **The submitted paper is smaller than the F1–F8 brief.** It contains exactly
   **5 figures** and **3 tables**. Several items in the brief (F4, F6, F7, the
   88-ECA appendix table T3, and the C8 nilpotency thread) do **not** appear in
   this submission. Per the brief's own conditional wording ("only if it appears
   in the submission"), these are out of scope for the *first submission* but
   are noted as optional/paper-value questions below.

4. **The int64-overflow hazard is real and present.**
   [`tests/manual_calculation_L_spectrum_rule150.ipynb`](tests/manual_calculation_L_spectrum_rule150.ipynb)
   calls `np.linalg.matrix_power(J, power)` on integer Jacobians (5 occurrences).
   This is exactly the artefact class the paper warns about, and it doubles as
   the raw material for the benchmark figure and the numerical-stability
   demonstration (C7).

5. **Nothing runs as-is in the current environment.** System `python3` is 3.9.6
   with none of numpy/scipy/igraph/torch/cellpylib installed. `environment.yml`
   pins **Windows-only conda builds** (e.g. `py311hd77b12b`), so it will not
   resolve cross-platform and is not a reproducible spec for this Mac / CI. A
   fresh, OS-neutral `requirements.txt` is required.

6. **The paper's `.bib` is missing.** `\bibliography{lyapunov-first_submission_CSF}`
   has no corresponding `.bib` file in `csf-submission/`. Out of scope for figure
   reproduction, but flagged.

---

## 2. Scope reconciliation — brief (F1–F8) vs submitted paper

| Brief item | In submission? | Manuscript object | File-stem in tex |
|---|---|---|---|
| F1 singular values + Lyapunov spectra, 16 affine ECAs, N~3001 | **Yes** (as *NO_CLASSES* variant, ordered by gradient weight) | Fig. `eca-spectra` | `singular_values_and_lyapunov_spectra_of_constant_J_ECAs_NO_CLASSES` |
| F2 2-D parity heatmaps + spectra (vN, Moore, r2-vN) | **Yes** | Fig. `2d-parity` | `singular_values_and_log_spectra_2d_parity` |
| F3 defect cones ECAs 32,4,30,54,150 (N=51,T=50) | **Partly** — appears as the single "difference patterns" figure | Fig. `defect-cones` | `persistent_defect_eca_diff` |
| F4 2-D parity defect reproduction (vN, Moore) | **No** | — | — |
| F5 defect propagation across topologies (Ring/Grid/WS/BA) | **Yes** | Fig. `defect-topologies` | `defect_propagation_networks_parity` |
| F6 Shannon entropy vs eigenvector centrality | **No** | — | — |
| F7 persistent defects, central cell clamped to 1 | **No** | — (tex has `_diff`, not `_clamped`) | — |
| F8 difference patterns (clamped) | **Partly** — the un-clamped `_diff` figure is Fig. `defect-cones` | Fig. `defect-cones` | `persistent_defect_eca_diff` |
| Benchmark (closed form vs Benettin) | **Yes** | Fig. `benchmark` | `benchmark_rule150` |
| T1 16 affine ECAs, gradients, classes | **Yes** | Tab. `affine-ecas` | — |
| T2 structure factor K(k,l) + parity MLE | **Yes** | Tab. `structure-factors` | — |
| T3 corrected 88-ECA gradient appendix | **Partly** — only the 4 corrected rules (62,110,130,146) are printed | Tab. `gradient-table` | — |

> **Note on F3/F8:** the submission has a *single* figure named
> `persistent_defect_eca_diff` (difference patterns). The exact rule list, `N`,
> `T` and seed for that figure are **not stated in the tex caption** — a
> paper-value question (see §8). The brief's F3 values (rules 32,4,30,54,150;
> N=51; T=50) are the assumed defaults pending confirmation.

---

## 3. Asset inventory

### 3a. Lyapunov-relevant code (in scope)

| Path | Purpose | Runs as-is? | Red flags |
|---|---|---|---|
| [`src/analysis.py`](src/analysis.py) (1153 ln) | Core maths. `jacobian_ECA` (Boolean derivative via `cellpylib`), `jacobian` (LLNA/torch), `calculate_Yt` (tangent-space direct-mult, float64), `lyapunov_spectrum` (log of SVD), `lyapunov_spectrum_analytical` (circulant closed form), `eca_has_constantJ` | No (needs env) | `eca_has_constantJ` is **hardcoded** to a 16-rule list (TODO admits it) — cannot serve as an independent C1 check. `lyapunov_spectrum_analytical` has a dead guard: `if np.any(eigenvalues) < 0` compares a bool to 0 → always False. Heavy deps: `torch`, `cellpylib`. |
| [`src/rules.py`](src/rules.py) (225 ln) | ECA/LLNA rule utilities: non-equivalent rules, L-R & B-W symmetry, `eca_as_binary`, `eca_to_llna`, `eca_is_totalistic` | No | No gradient/Vichniac computation present (needed for C4/T3). |
| [`src/automata.py`](src/automata.py) (363 ln) | `LLNA` torch module: interval encoding, `step`, diagram | No | torch/torch_geometric. Overkill for the parity/affine paper. |
| [`src/networks.py`](src/networks.py) (65 ln) | `create_2d_torus_lattice`, `watts_strogatz_rewire` (igraph) | No | Useful for F5 topologies (Grid, WS). |
| [`lyapunov_exp/lyapunov.py`](lyapunov_exp/lyapunov.py) (64 ln) | Rough script: `jacobian`, `conf_space`, `tangent_space` | **No** — undefined `N`, `LLNA`, `np`, `tc` at top; `s0 ^ s0_defect` XORs float arrays (TypeError) | Dead/experimental. Superseded by `src/analysis.py`. Do not reuse. |
| [`tests/manual_calculation_L_spectrum_rule150.ipynb`](tests/manual_calculation_L_spectrum_rule150.ipynb) (40 cells) | Benchmark raw material: `svd_rule150`, `circulant_matrix`, `singular_values_from_rule`, `eca_complement/mirror`, direct-mult at float32 & float64, `matrix_power` | Unknown | **int64 overflow** (`matrix_power` on int J). No Benettin QR present. Paper claims float16+float64+Benettin; notebook has float32+float64 only. |
| [`tests/sanity-check/lyapunov.ipynb`](tests/sanity-check/lyapunov.ipynb), [`llna.ipynb`](tests/sanity-check/llna.ipynb) | Sanity checks (no figures saved) | Unknown | Reference material only. |
| [`tests/manual_calculation…`, `tests/*.ipynb`] misc | Exploratory (Derrida, defect tracing, etc.) | Unknown | Mostly out of scope. |

### 3b. Out-of-scope assets (the *other* paper — do not import)

- `manuscript/*.ipynb` (essential_metrics, impact_analysis, lllna_robustness) →
  write to a `figures/` dir (gitignored) for the robustness paper.
- `manuscript/data/**` (`.npy`: final-state densities, FSSP success rates,
  totalitarian-rule impact) — robustness-paper data.
- `presentation/`, `overview.ipynb`, `1-create_nets.py`, `2-run_automata.py`,
  `src/datasets.py`, `src/simulation.py`, `src/network_analysis` (git submodule),
  `network_automata_robustness/` (stray conda `.dll.conda_trash` files),
  `tests/*.jl`, `et` (stray file).

### 3c. Data & figures

- **No Lyapunov-paper data files** exist (graphs, configs). F5's WS/BA graphs
  must be generated deterministically from seeds — a gap.
- **No committed figures**: `figures/` is gitignored. The only rendered copies
  of the 5 target figures are embedded in
  [`csf-submission/lyapunov-first_submission_CSF.pdf`](csf-submission/lyapunov-first_submission_CSF.pdf),
  which is the visual-comparison reference for Phase 2.

---

## 4. Red-flag register

| # | Flag | Location | Impact |
|---|---|---|---|
| R1 | int64 overflow via `matrix_power` on integer J | benchmark notebook (×5) | Wrong spectra for moderate T; **the paper's headline artefact**. Fix: GF(2) / Python-int / object dtype / mod-each-step. Keep central + tested. |
| R2 | `eca_has_constantJ` hardcoded | `analysis.py:960` | Cannot be used to *prove* C1; C1 needs first-principles recompute. |
| R3 | Dead negativity guard | `analysis.py:763` | Silent; hides genuine negative eigenvalues. |
| R4 | Windows-only conda pins | `environment.yml` | Not reproducible cross-platform/CI. Replace with pinned `requirements.txt`. |
| R5 | Nondeterminism | `lyapunov.py` (`np.random` no seed); F5 graph generators | Two runs differ. All seeds must be fixed + documented. |
| R6 | Broken script | `lyapunov_exp/lyapunov.py` | Does not import/run; float XOR bug. Exclude. |
| R7 | Missing `.bib` | `csf-submission/` | Paper won't compile refs (out of scope for figures). |
| R8 | No Benettin QR implementation | anywhere | Needed for the benchmark (C2/Fig 3) and C6. Must be written. |
| R9 | float16 direct-mult absent | benchmark notebook | Paper shows 16-bit; notebook has float32. Confirm intended precision. |

---

## 5. Figure → best existing asset (canonical source)

| Fig | Canonical maths source to reuse | Status |
|---|---|---|
| eca-spectra (F1) | `lyapunov_spectrum_analytical` + Eq. (8) closed form | Reimplement cleanly from Eq. (8); reference impl exists |
| 2d-parity (F2) | Eq. (14) structure factor `K(k,l)` | **No code** — from formula |
| benchmark (F3/Fig) | notebook `svd_rule150`, `circulant_matrix`, direct-mult | Partial; **Benettin missing** |
| defect-cones (persistent_defect_eca_diff) | ECA evolution + Boolean diff (cellpylib or pure numpy) | **No dedicated script** |
| defect-topologies (F5) | `A^t e_j mod 2`; `networks.py` for Grid/WS | **No dedicated script**; needs seeded WS/BA + eigenvector-centrality ordering |

## 6. Claim → verification asset

| Claim | Existing support | Gap |
|---|---|---|
| C1 16 affine ECAs = constant-J rules | hardcoded list only (R2) | Need first-principles recompute of all 256 |
| C2 closed form == Benettin | closed form yes; **Benettin no** (R8) | Write Benettin QR |
| C3 MLE values (ln3, ln2; ln5/9/13; 2ln3) | formulas in paper | Write direct numeric checks |
| C4 recompute 88 gradients; confirm 4 corrections | **no gradient code** | Write `verify_vichniac.py` (gradient + Quine–McCluskey + CSV) |
| C5 parity MLE=ln ρ(A); amplitude ∝ eigvec centrality | `networks.py` partial | Write numeric check on seeded WS/BA |
| C6 Σ exponents = ln|det J| | none | Depends on Benettin (R8) |
| C7 numerical-artefact demo | notebook `matrix_power` (R1) | Package as neutral side-by-side |
| C8 nilpotency (90/165, 60/102/195/153) | **not in submission** | Out of scope unless thread added (see top-of-tex open comment) |

---

## 7. Proposed target structure (`paper-repo/`)

Lean core: the Lyapunov/parity paper needs only numpy/scipy + a graph library +
matplotlib. **Drop torch, torch_geometric and cellpylib** (used only by the LLNA
robustness paper); reimplement ECA evolution and the Boolean Jacobian in pure
numpy so the core is small, fast and int64-safe.

```
paper-repo/
  README.md                overview, install, one-command reproduce, figure/claim map
  CLAUDE.md                orientation for future sessions
  LICENSE                  MIT (default)
  CITATION.cff             cite paper + repo
  requirements.txt         exact pinned, OS-neutral versions
  pyproject.toml           make src/lyapunov importable
  reproduce.py             `all` (everything) | `quick` (CI subset)
  src/lyapunov/
    gf2.py                 GF(2) / int64-safe matrix power (the central fix)
    rules.py               ECA rule table → Boolean gradient; affine detection; 88-rule enumeration
    jacobian.py            constant Jacobian (circulant / adjacency), pure-numpy ECA step
    spectra.py             closed-form circulant & multilevel-circulant (DFT) singular values
    benettin.py            Benettin QR + direct-multiplication (float16/32/64) reference methods
    parity.py              parity rule on a graph; A^t e_j mod 2; eigenvector centrality
    quine_mccluskey.py     Boolean simplifier for the gradient table
  figures/
    fig_eca_spectra.py             -> singular_values_and_lyapunov_spectra_of_constant_J_ECAs_NO_CLASSES
    fig_2d_parity.py               -> singular_values_and_log_spectra_2d_parity
    fig_benchmark.py               -> benchmark_rule150   (make_benchmark_figure.py aliasable)
    fig_defect_cones.py            -> persistent_defect_eca_diff
    fig_defect_topologies.py       -> defect_propagation_networks_parity
  verification/
    test_c1_affine_constant_jacobian.py
    test_c2_benchmark_closed_form_vs_benettin.py
    test_c3_mle_values.py
    test_c4_vichniac_gradients.py         (+ verify_vichniac.py CLI, CSV export)
    test_c5_parity_centrality.py
    test_c6_benettin_det_sum.py
    test_c7_numerical_artefact.py
  data/
    make_graphs.py         regenerate seeded WS/BA graphs deterministically
    tables/                generated CSVs (T1, T2, T3/gradient table)
  output/                  generated figures  (recommend: gitignore, keep .gitkeep)
  docs/provenance.md       figure/claim -> script -> command -> output -> status
  .github/workflows/ci.yml (optional) runs `reproduce.py quick`
```

---

## 8. Gap list (needs code, a seed, or a manuscript value)

**Must build (no working source in repo):**
- G1 `src/lyapunov` pure-numpy core (int64-safe `gf2` power; the central fix).
- G2 Benettin QR algorithm (R8) — blocks benchmark, C2, C6.
- G3 `verify_vichniac.py`: rule-table → Boolean gradient for all 88 ECAs, +
  Quine–McCluskey simplifier, + CSV export; confirm the 4 corrections (C4/T3).
- G4 First-principles affine/constant-J detector for all 256 (C1) — replace R2.
- G5 Five figure scripts (none exist).
- G6 Deterministic graph generation for F5 (seeded WS N=200 k=6 p=0.2; BA N=200 m=3),
  with eigenvector-centrality node ordering (C5).
- G7 OS-neutral pinned `requirements.txt` (replace Windows conda `environment.yml`).

**Manuscript values needed from author (cannot be derived):**
- Q1 `persistent_defect_eca_diff` (Fig. defect-cones): exact rule list, N, T,
  and RNG seed. (Brief F3 suggests 32,4,30,54,150 / N=51 / T=50 — confirm.)
- Q2 Benchmark precisions: paper says "16-bit and 64-bit"; notebook uses
  float32/float64. Confirm float16 is intended (R9).
- Q3 F5 seeds: exact RNG seeds used for the published WS and BA graphs (so
  regenerated graphs match the figure), or accept new fixed seeds.
- Q4 Fig. eca-spectra is the `NO_CLASSES` variant (ordered by gradient weight);
  confirm this is canonical (brief F1 said "grouped by class").

**Out of scope for first submission (confirm to exclude):**
- F4, F6, F7; the full 88-row appendix table (only 4 rows printed); C8 nilpotency
  (only an open-question comment at the top of the tex).
```
