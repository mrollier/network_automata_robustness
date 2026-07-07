"""Functional core of the LLNA dynamics.

`neighbor_density` replaces the former torch_geometric
`SimpleConv(aggr="mean")` message passing. States are exactly 0/1, so the
neighbour sum is a small integer — exactly representable in float32 — and
sum-then-divide reproduces the message-passing mean bit-for-bit (gated by
the characterization fixtures). This removed the torch_geometric dependency
and works on cpu, mps, and cuda.

`interval_index` mirrors `LLNA.interval_encoding` predicate-for-predicate
(same float32 comparisons), returning integer interval indices instead of a
one-hot tensor; `step_batch` uses it to evolve K rules in lockstep without
materialising a [K, L, N, resolution] one-hot.
"""

import torch as tc


class GraphConnectivity:
    """Precomputed edge structure for repeated density evaluations.

    Parameters
    ----------
    edge_index : torch.Tensor
        Long tensor [2, M] of directed edges (both directions present for
        undirected graphs, as everywhere in this package).
    num_nodes : int, optional
        Defaults to max node id + 1.
    """

    def __init__(self, edge_index: tc.Tensor, num_nodes: int | None = None):
        edge_index = edge_index.long()
        self.src = edge_index[0]
        self.dst = edge_index[1]
        self.num_nodes = int(edge_index.max().item()) + 1 if num_nodes is None else num_nodes
        in_degree = tc.zeros(self.num_nodes, dtype=tc.float32, device=edge_index.device)
        in_degree.index_add_(
            0, self.dst, tc.ones(self.dst.numel(), dtype=tc.float32, device=edge_index.device)
        )
        # isolated nodes get density 0 (the message-passing mean convention)
        self.safe_degree = in_degree.clamp(min=1.0)

    def to(self, device) -> "GraphConnectivity":
        clone = object.__new__(GraphConnectivity)
        clone.src = self.src.to(device)
        clone.dst = self.dst.to(device)
        clone.num_nodes = self.num_nodes
        clone.safe_degree = self.safe_degree.to(device)
        return clone


def neighbor_density(h: tc.Tensor, conn) -> tc.Tensor:
    """Mean state of each node's in-neighbours.

    Parameters
    ----------
    h : torch.Tensor
        States, shape [..., N], any numeric dtype with exact 0/1 values.
    conn : GraphConnectivity or torch.Tensor
        Precomputed connectivity, or a raw [2, M] edge tensor (the node count
        is then taken from h, matching the old message-passing behaviour).

    Returns
    -------
    torch.Tensor
        float32 densities, shape [..., N].
    """
    hf = tc.atleast_2d(h).to(tc.float32)
    if not isinstance(conn, GraphConnectivity):
        conn = GraphConnectivity(conn, num_nodes=hf.shape[-1])
    total = tc.zeros(hf.shape[:-1] + (conn.num_nodes,), dtype=tc.float32, device=hf.device)
    total.index_add_(-1, conn.dst, hf[..., conn.src])
    return total / conn.safe_degree


def interval_index(p: tc.Tensor, resolution: int, iso: bool) -> tc.Tensor:
    """Integer interval index of each density; identical comparison semantics
    to LLNA.interval_encoding (argmax of the one-hot), without materialising
    the one-hot."""
    idx = tc.zeros_like(p, dtype=tc.long)
    if not iso:  # classic pseudo-isomorphic case
        rp = resolution * p
        for k in range(1, resolution):
            idx = tc.where((k <= rp) & (rp < k + 1), k, idx)
        idx = tc.where(p == 1, resolution - 1, idx)
    else:  # altered isomorphic case
        for k in range(1, resolution):
            if k == (resolution - 1) / 2:  # middle interval: closed on both sides
                mask = (p >= k / resolution) & (p <= (k + 1) / resolution)
            elif k < resolution / 2:
                mask = (p >= k / resolution) & (p < (k + 1) / resolution)
            else:
                mask = (p > k / resolution) & (p <= (k + 1) / resolution)
            idx = tc.where(mask, k, idx)
    return idx


def rule_tables(rules, resolution: int, device="cpu") -> tuple[tc.Tensor, tc.Tensor]:
    """One row per rule: born/survive response per interval.

    Parameters
    ----------
    rules : array-like [K, 2]
        Integer (beta, sigma) pairs.

    Returns
    -------
    (born, survive) : float32 tensors [K, resolution]
    """
    rules = tc.as_tensor(rules, dtype=tc.long, device=device)
    bits = tc.arange(resolution, device=device)
    born = ((rules[:, 0:1] >> bits) & 1).to(tc.float32)
    survive = ((rules[:, 1:2] >> bits) & 1).to(tc.float32)
    return born, survive


def step_batch(
    h: tc.Tensor, conn, born: tc.Tensor, survive: tc.Tensor, resolution: int, iso: bool
) -> tc.Tensor:
    """One synchronous update of K rules in lockstep.

    Parameters
    ----------
    h : torch.Tensor
        States [K, L, N] (or [L, N] with K==1 tables), exact 0/1 values.
    born, survive : torch.Tensor
        Rule tables [K, resolution] from rule_tables().

    Returns
    -------
    torch.Tensor
        Next states, float32, same shape as h.
    """
    p = neighbor_density(h, conn)
    idx = interval_index(p, resolution, iso)
    k_index = tc.arange(born.shape[0], device=idx.device).view(-1, *([1] * (idx.dim() - 1)))
    b = born[k_index, idx]
    s = survive[k_index, idx]
    hf = h.to(tc.float32)
    return b * (1 - hf) + s * hf
