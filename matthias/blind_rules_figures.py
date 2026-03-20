"""
Publication-quality figures of 1D cellular automata space-time diagrams
showing the effect of "blind rules" (zealous nodes) on elementary cellular
automata (ECA).

Figure 1: Phase transitions under blind-rule perturbation (null rule, ECA 0)
Figure 2: Class III vs Class IV fragility comparison (identity rule, ECA 204)
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Style ──────────────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "cm",
    "figure.dpi": 100,
    "savefig.dpi": 1200,
    "font.size": 11,
})

OUT_DIR = Path(__file__).parent


# ── Core simulation ────────────────────────────────────────────────────────────

def make_rule_table(rule_number: int) -> np.ndarray:
    """Return an 8-element lookup table for the given ECA rule number."""
    return np.array([(rule_number >> i) & 1 for i in range(8)], dtype=np.uint8)


def simulate_nu_eca(
    rule_main: int,
    rule_blind: int,
    nu: float,
    grid_size: int = 400,
    timesteps: int = 400,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate a non-uniform ECA (ν-ECA).

    Parameters
    ----------
    rule_main  : ECA rule number for the main population of cells.
    rule_blind : ECA rule number for the 'blind' cells.
    nu         : Fraction of cells governed by the blind rule (0 ≤ ν ≤ 1).
                 ν=0 gives a regular ECA; ν=1 gives a fully blind automaton.
    grid_size  : Number of cells.
    timesteps  : Number of time steps (rows in output, including t=0).
    seed       : NumPy random seed for reproducibility.

    Returns
    -------
    history : (timesteps, grid_size) uint8 array, history[0] is the IC.
    """
    rng = np.random.default_rng(seed)

    # Assign each cell to a rule: True → blind rule, False → main rule
    is_blind = rng.random(grid_size) < nu

    table_main  = make_rule_table(rule_main)
    table_blind = make_rule_table(rule_blind)

    # Random initial condition
    state = rng.integers(0, 2, size=grid_size, dtype=np.uint8)

    history = np.empty((timesteps, grid_size), dtype=np.uint8)
    history[0] = state

    for t in range(1, timesteps):
        # Vectorised neighbourhood index: i = 4*left + 2*centre + right
        left   = np.roll(state, 1)
        right  = np.roll(state, -1)
        idx    = (left.astype(np.int32) * 4
                  + state.astype(np.int32) * 2
                  + right.astype(np.int32))

        new_state = np.where(is_blind, table_blind[idx], table_main[idx])
        state = new_state.astype(np.uint8)
        history[t] = state

    return history


# ── Rule verification ──────────────────────────────────────────────────────────

def _verify_rules():
    """Quick sanity check on the two blind rules used."""
    null_table     = make_rule_table(0)
    identity_table = make_rule_table(204)
    inversion_table = make_rule_table(51)

    assert np.all(null_table == 0), "Rule 0 must output 0 for all inputs"

    # Identity: output = centre cell (bit 1 of neighbourhood index)
    for i in range(8):
        centre = (i >> 1) & 1
        assert identity_table[i] == centre, f"Rule 204 failed at index {i}"

    # Inversion: output = NOT centre cell
    for i in range(8):
        centre = (i >> 1) & 1
        assert inversion_table[i] == (1 - centre), f"Rule 51 failed at index {i}"


_verify_rules()


# ── Figure helpers ─────────────────────────────────────────────────────────────

def _show_spacetime(ax, history, title=''):
    ax.imshow(
        history,
        cmap='binary',
        interpolation='none',
        aspect='equal',
        origin='upper',
    )
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=11, pad=3)


# ── Figure 1 ───────────────────────────────────────────────────────────────────

def make_figure1(out_dir: Path):
    """1×4 panel: null-rule perturbation, phase transitions."""

    panels = [
        dict(rule_main=30,  nu=0.00, seed=42,  title=r'Rule 30, $\nu=0.00$'),
        dict(rule_main=30,  nu=0.01, seed=42,  title=r'Rule 30, $\nu=0.01$'),
        dict(rule_main=30,  nu=0.10, seed=42,  title=r'Rule 30, $\nu=0.10$'),
        dict(rule_main=106, nu=0.01, seed=42,  title=r'Rule 106, $\nu=0.01$'),
    ]

    fig, axes = plt.subplots(
        1, 4,
        figsize=(6.5, 3.0),
        layout='constrained',
        gridspec_kw=dict(wspace=0.04, hspace=0),
    )

    for ax, p in zip(axes, panels):
        history = simulate_nu_eca(
            rule_main=p['rule_main'],
            rule_blind=0,           # null rule
            nu=p['nu'],
            grid_size=100,
            timesteps=150,
            seed=p['seed'],
        )
        _show_spacetime(ax, history, title=str(p['title']))

    axes[0].set_ylabel(r'$\leftarrow$ time', fontsize=12, labelpad=4)



    for ext in ('pdf', 'png'):
        fname = out_dir / f'figure_blind_rules_phase_transition.{ext}'
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        print(f'Saved {fname}')

    plt.close(fig)


# ── Figure 2 ───────────────────────────────────────────────────────────────────

def make_figure2(out_dir: Path):
    """2×4 panel: Class IV (top) vs Class III (bottom) with identity rule."""

    class_iv_rules  = [ 54, 106, 110, 124]
    class_iii_rules = [ 30,  45,  90, 150]

    fig, axes = plt.subplots(
        2, 4,
        figsize=(6.5, 6.0),
        layout='constrained',
        gridspec_kw=dict(wspace=0.04, hspace=0.01),
    )

    for col, rule in enumerate(class_iv_rules):
        history = simulate_nu_eca(
            rule_main=rule,
            rule_blind=204,         # identity rule
            nu=0.01,
            grid_size=100,
            timesteps=150,
            seed=42,
        )
        _show_spacetime(axes[0, col], history, title=f'Rule {rule}')

    for col, rule in enumerate(class_iii_rules):
        history = simulate_nu_eca(
            rule_main=rule,
            rule_blind=204,         # identity rule
            nu=0.01,
            grid_size=100,
            timesteps=150,
            seed=42,
        )
        _show_spacetime(axes[1, col], history, title=f'Rule {rule}')

    # Row labels on the left
    for row_idx, row_label in enumerate(['Class IV', 'Class III']):
        axes[row_idx, 0].set_ylabel(row_label, fontsize=12, labelpad=6)



    for ext in ('pdf', 'png'):
        fname = out_dir / f'figure_blind_rules_class_comparison.{ext}'
        fig.savefig(fname, dpi=300, bbox_inches='tight')
        print(f'Saved {fname}')

    plt.close(fig)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    make_figure1(OUT_DIR)
    make_figure2(OUT_DIR)
    print('Done.')
