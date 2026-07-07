"""Batched rule sweeps: evolve many rules in lockstep over shared graphs and
initial configurations.

The rule dimension K is the leading tensor axis; each step computes neighbour
densities and interval indices for all rules at once and gathers the response
from [K, resolution] rule tables (llna.engine.step_batch). Values are exactly
equal to running llna.automata.LLNA rule-by-rule (gated by tests/test_sweep.py);
the win is one vectorised kernel per step instead of K Python-level model calls.
"""

from collections.abc import Sequence

import numpy as np
import torch as tc
from tqdm import tqdm

from llna.engine import GraphConnectivity, rule_tables, step_batch


def auto_rule_chunk(
    num_rules: int, num_configs: int, num_nodes: int, budget_bytes: int = 256 * 1024**2
) -> int:
    """Largest rule-chunk size whose float32 working set (state tensor plus
    ~3 step temporaries) stays under budget_bytes."""
    per_rule = max(num_configs * num_nodes * 4 * 4, 1)
    return int(max(1, min(num_rules, budget_bytes // per_rule)))


def evolve_rules_batch(
    edge_index,
    init_configs,
    rules: Sequence[tuple[int, int]] | np.ndarray,
    resolution: int,
    iso: bool = True,
    T: int = 100,
    device: str = "cpu",
    rule_chunk: int | None = None,
    reduce: str = "node_mean",
    progress: bool = False,
) -> np.ndarray:
    """Evolve K rules over the same graph and initial configurations.

    Parameters
    ----------
    edge_index : array-like [2, M]
        Directed edges (both directions for undirected graphs).
    init_configs : array-like [L, N]
        Initial configurations with exact 0/1 values.
    rules : array-like [K, 2]
        Integer (beta, sigma) pairs.
    T : int
        Number of steps; trajectories include the initial configuration.
    device : str
        "cpu", "mps", or "cuda". Results are identical across devices
        (all arithmetic is exact in float32 for 0/1 states).
    rule_chunk : int, optional
        Rules per batch; default keeps the working set under ~256 MB.
    reduce : str
        "node_mean" -> float32 [K, L, T+1] mean state per timestep;
        "none"      -> uint8   [K, L, T+1, N] full trajectories (memory!).

    Returns
    -------
    numpy.ndarray
        See `reduce`.
    """
    if reduce not in ("node_mean", "none"):
        raise ValueError(f"Unknown reduce mode: {reduce!r}")
    dev = tc.device(device)
    edge_index = tc.as_tensor(np.asarray(edge_index), dtype=tc.long)
    init_configs = tc.as_tensor(np.asarray(init_configs), dtype=tc.float32)
    if init_configs.dim() != 2:
        raise ValueError("init_configs must be [L, N].")
    L, N = init_configs.shape
    rules = np.asarray(rules, dtype=np.int64)
    K = len(rules)

    conn = GraphConnectivity(edge_index, num_nodes=N).to(dev)
    init_configs = init_configs.to(dev)
    if rule_chunk is None:
        rule_chunk = auto_rule_chunk(K, L, N)

    outputs = []
    chunk_starts = range(0, K, rule_chunk)
    if progress:
        chunk_starts = tqdm(chunk_starts, desc=f"rule chunks (x{rule_chunk})")
    for start in chunk_starts:
        chunk = rules[start : start + rule_chunk]
        k = len(chunk)
        born, survive = rule_tables(chunk, resolution, device=dev)
        h = init_configs.unsqueeze(0).expand(k, L, N).clone()
        if reduce == "node_mean":
            out = tc.empty((k, L, T + 1), dtype=tc.float32, device=dev)
            out[:, :, 0] = h.mean(dim=-1)
        else:
            out = tc.empty((k, L, T + 1, N), dtype=tc.uint8, device=dev)
            out[:, :, 0] = h.to(tc.uint8)
        for t in range(1, T + 1):
            h = step_batch(h, conn, born, survive, resolution, iso)
            if reduce == "node_mean":
                out[:, :, t] = h.mean(dim=-1)
            else:
                out[:, :, t] = h.to(tc.uint8)
        outputs.append(out.cpu().numpy())
    return np.concatenate(outputs, axis=0)


def evolve_defect_pairs(
    edge_index,
    init_configs,
    defected_configs,
    rules: Sequence[tuple[int, int]] | np.ndarray,
    resolution: int,
    iso: bool = True,
    T: int = 100,
    device: str = "cpu",
    rule_chunk: int | None = None,
    progress: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Evolve clean and defected twins of each configuration in lockstep.

    Returns
    -------
    (state_means, defect_means) : float32 arrays [K, L, T+1]
        Per-step node-mean of the clean state and of the clean-vs-defected
        disagreement (the defect density) — the state/defect convergence
        workload of the manuscript. Memory stays O(K*L*N) because the XOR
        reduction happens per step instead of on stored trajectories.
    """
    dev = tc.device(device)
    edge_index = tc.as_tensor(np.asarray(edge_index), dtype=tc.long)
    clean0 = tc.as_tensor(np.asarray(init_configs), dtype=tc.float32)
    defect0 = tc.as_tensor(np.asarray(defected_configs), dtype=tc.float32)
    if clean0.shape != defect0.shape or clean0.dim() != 2:
        raise ValueError("init_configs and defected_configs must both be [L, N].")
    L, N = clean0.shape
    rules = np.asarray(rules, dtype=np.int64)
    K = len(rules)

    conn = GraphConnectivity(edge_index, num_nodes=N).to(dev)
    clean0, defect0 = clean0.to(dev), defect0.to(dev)
    if rule_chunk is None:
        rule_chunk = auto_rule_chunk(K, 2 * L, N)

    state_out, defect_out = [], []
    chunk_starts = range(0, K, rule_chunk)
    if progress:
        chunk_starts = tqdm(chunk_starts, desc=f"rule chunks (x{rule_chunk})")
    for start in chunk_starts:
        chunk = rules[start : start + rule_chunk]
        k = len(chunk)
        born, survive = rule_tables(chunk, resolution, device=dev)
        h_clean = clean0.unsqueeze(0).expand(k, L, N).clone()
        h_defect = defect0.unsqueeze(0).expand(k, L, N).clone()
        states = tc.empty((k, L, T + 1), dtype=tc.float32, device=dev)
        defects = tc.empty((k, L, T + 1), dtype=tc.float32, device=dev)
        states[:, :, 0] = h_clean.mean(dim=-1)
        defects[:, :, 0] = (h_clean != h_defect).to(tc.float32).mean(dim=-1)
        for t in range(1, T + 1):
            h_clean = step_batch(h_clean, conn, born, survive, resolution, iso)
            h_defect = step_batch(h_defect, conn, born, survive, resolution, iso)
            states[:, :, t] = h_clean.mean(dim=-1)
            defects[:, :, t] = (h_clean != h_defect).to(tc.float32).mean(dim=-1)
        state_out.append(states.cpu().numpy())
        defect_out.append(defects.cpu().numpy())
    return np.concatenate(state_out, axis=0), np.concatenate(defect_out, axis=0)
