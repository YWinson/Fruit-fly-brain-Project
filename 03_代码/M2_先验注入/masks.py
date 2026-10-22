"""Mask construction and null models for M2.

The biological mask is a 64 x 32 binary matrix. The two null arms must match it
in different invariants:

* ``deg_rewired``: bipartite degree-preserving edge swaps (row and column sums
  exactly preserved, no duplicate edges).
* ``er_random``: uniformly random E edges (only the edge count is preserved).

The draft's danger point ① is a silent degree-preservation failure; every
rewiring call therefore runs hard assertions and raises on violation.
"""

from __future__ import annotations

import numpy as np
import networkx as nx

from .clusters import GranularityResult, expand_cluster


class MaskError(AssertionError):
    """Raised when a mask or null violates a structural invariant."""


def mask_stats(mask: np.ndarray) -> dict:
    mask = np.asarray(mask, dtype=bool)
    out_degree = mask.sum(axis=1)
    in_degree = mask.sum(axis=0)
    return {
        "n_edges": int(mask.sum()),
        "density": float(mask.mean()),
        "out_degree": out_degree.astype(int).tolist(),
        "in_degree": in_degree.astype(int).tolist(),
        "out_degree_sequence": sorted(out_degree.astype(int).tolist(), reverse=True),
        "in_degree_sequence": sorted(in_degree.astype(int).tolist(), reverse=True),
        "n_out_nodes_nonzero": int((out_degree > 0).sum()),
        "n_in_nodes_nonzero": int((in_degree > 0).sum()),
    }


def top_k_mask(weights: np.ndarray, density: float) -> np.ndarray:
    """Keep the globally strongest ``density`` fraction of the 64x32 entries."""
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 2:
        raise ValueError(f"weights must be 2-D, got {weights.shape}")
    if not 0.0 < density <= 1.0:
        raise ValueError(f"density must be in (0, 1], got {density}")

    n_total = weights.size
    n_keep = max(1, int(round(density * n_total)))
    flat = weights.ravel()
    positive_indices = np.flatnonzero(flat > 0)
    if positive_indices.size == 0:
        raise MaskError("connectivity matrix has no positive entries")

    if positive_indices.size <= n_keep:
        selected = positive_indices
    else:
        top = positive_indices[np.argsort(flat[positive_indices])[-n_keep:]]
        selected = top

    mask = np.zeros(n_total, dtype=bool)
    mask[selected] = True
    return mask.reshape(weights.shape)


def balanced_topk_mask(weights: np.ndarray, density: float) -> np.ndarray:
    """Row-balanced top-k mask.

    The globally strongest 10% of a biological matrix often forms a nested
    (Ferrers-like) graph: a few hub rows dominate and the degree sequence then
    forces almost every edge, so a degree-preserving null cannot move enough
    edges (the draft's danger point ①). Taking the strongest entries
    round-robin across rows keeps the density and the "strongest per row"
    principle, but yields a non-degenerate degree sequence that the null can
    actually randomise.
    """
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 2:
        raise ValueError(f"weights must be 2-D, got {weights.shape}")
    n_total = weights.size
    n_keep = max(1, int(round(density * n_total)))

    mask = np.zeros_like(weights, dtype=bool)
    rows = [np.argsort(weights[i][weights[i] > 0])[::-1] for i in range(weights.shape[0])]
    depth = 0
    while int(mask.sum()) < n_keep:
        added = False
        for row_index, columns in enumerate(rows):
            if depth < len(columns):
                mask[row_index, columns[depth]] = True
                added = True
                if int(mask.sum()) >= n_keep:
                    break
        if not added:
            break
        depth += 1
    return mask


def sign_from_cluster(result: GranularityResult) -> np.ndarray:
    """Sign matrix (+1 excitatory, -1 inhibitory) expanded to 64x32."""
    signed = result.W_exc_expanded - result.W_inh_expanded
    sign = np.where(signed >= 0, 1, -1).astype(np.int8)
    # Where there is no connection the sign is irrelevant; keep +1 for clarity.
    sign[result.W_syn_expanded <= 0] = 1
    return sign


def build_bio_mask(
    result: GranularityResult,
    density: float = 0.10,
    selection: str = "balanced",
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build ``M_bio`` and ``Sigma`` from a granularity result.

    ``selection='balanced'`` (default) takes the strongest entries round-robin
    across rows; ``selection='global'`` is the literal global top-k variant.
    The balanced variant is required for the draft's degree-preserving null to
    move at least E edges (danger point ①).
    """
    if selection == "balanced":
        mask = balanced_topk_mask(result.W_syn_expanded, density=density)
    elif selection == "global":
        mask = top_k_mask(result.W_syn_expanded, density=density)
    else:
        raise ValueError(f"unknown selection: {selection!r}")

    sign = sign_from_cluster(result)
    info = mask_stats(mask)
    info.update(
        {
            "granularity": result.name,
            "k_out": result.k_out,
            "k_in": result.k_in,
            "requested_density": density,
            "selection": selection,
            "n_positive_expanded": int((result.W_syn_expanded > 0).sum()),
            "mask_id": f"bio_{result.name}_{selection}_p{density:g}",
        }
    )
    return mask, sign, info


def er_random_mask(
    n_out: int,
    n_in: int,
    n_edges: int,
    seed: int,
) -> np.ndarray:
    """Uniformly sample ``n_edges`` distinct edges from an n_out x n_in grid."""
    n_total = n_out * n_in
    if not 0 <= n_edges <= n_total:
        raise ValueError(f"n_edges={n_edges} outside [0, {n_total}]")
    rng = np.random.default_rng(int(seed))
    selected = rng.choice(n_total, size=n_edges, replace=False)
    mask = np.zeros(n_total, dtype=bool)
    mask[selected] = True
    return mask.reshape(n_out, n_in)


def dense_mask(n_out: int, n_in: int) -> np.ndarray:
    return np.ones((n_out, n_in), dtype=bool)


def _validate_bipartite_degrees(
    original: np.ndarray,
    rewired: np.ndarray,
) -> None:
    original = np.asarray(original, dtype=bool)
    rewired = np.asarray(rewired, dtype=bool)
    if original.shape != rewired.shape:
        raise MaskError(f"shape mismatch: {original.shape} vs {rewired.shape}")
    if not np.array_equal(original.sum(axis=1), rewired.sum(axis=1)):
        raise MaskError("row (out-degree) sequences differ after rewiring")
    if not np.array_equal(original.sum(axis=0), rewired.sum(axis=0)):
        raise MaskError("column (in-degree) sequences differ after rewiring")

    # NetworkX validation, as required by the draft's library rule.
    n_out, n_in = original.shape
    graph = nx.Graph()
    graph.add_nodes_from([("out", i) for i in range(n_out)], bipartite=0)
    graph.add_nodes_from([("in", j) for j in range(n_in)], bipartite=1)
    graph.add_edges_from(
        [("out", i), ("in", j)] for i, j in np.argwhere(rewired)
    )
    out_degrees = [graph.degree[("out", i)] for i in range(n_out)]
    in_degrees = [graph.degree[("in", j)] for j in range(n_in)]
    if out_degrees != rewired.sum(axis=1).tolist():
        raise MaskError("networkx out-degree validation failed")
    if in_degrees != rewired.sum(axis=0).tolist():
        raise MaskError("networkx in-degree validation failed")


def _curveball_rewire(
    mask: np.ndarray,
    seed: int,
    n_steps_factor: int = 20,
) -> tuple[np.ndarray, int]:
    """Curveball randomisation of a bipartite graph.

    For two rows i, j, let A be the columns only in i and B the columns only
    in j. Reassign the union A∪B randomly so that i keeps |A| of them. This
    preserves every row and column sum exactly and never creates duplicate
    edges. Curveball moves many more edges than naive pairwise swaps when
    the degree sequence contains many degree-0/1 nodes (our 10%-dense mask).
    """
    mask = np.asarray(mask, dtype=bool)
    rows = [set(np.flatnonzero(mask[i]).tolist()) for i in range(mask.shape[0])]
    rng = np.random.default_rng(int(seed))
    n_steps = max(200, int(n_steps_factor * mask.sum()))
    n_moved = 0

    for _ in range(n_steps):
        i, j = rng.choice(mask.shape[0], size=2, replace=False)
        a = rows[i] - rows[j]
        b = rows[j] - rows[i]
        if not a and not b:
            continue
        union = list(a | b)
        rng.shuffle(union)
        new_i = (rows[i] - a) | set(union[: len(a)])
        new_j = (rows[j] - b) | set(union[len(a) :])
        n_moved += len(new_i - rows[i]) + len(new_j - rows[j])
        rows[i], rows[j] = new_i, new_j

    rewired = np.zeros_like(mask, dtype=bool)
    for i, columns in enumerate(rows):
        if columns:
            rewired[i, list(columns)] = True
    return rewired, n_moved


def bipartite_degree_preserving_rewire(
    mask: np.ndarray,
    seed: int,
    n_swaps_factor: int = 20,
) -> tuple[np.ndarray, dict]:
    """Randomise a bipartite mask while preserving every row/column sum.

    The draft calls for a ``networkx``-based degree-preserving exchange. We
    use the curveball algorithm for the actual move set because the mask is
    very sparse (10% density: many rows have degree 0/1) and naive pairwise
    edge swaps stall far short of the required Hamming distance. NetworkX is
    still used below for an independent degree validation, and the
    ``n_swaps_factor`` name is kept for config compatibility.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got {mask.shape}")
    n_edges = int(mask.sum())
    if n_edges < 2:
        return mask.copy(), {"n_edges": n_edges, "n_moves": 0, "hamming": 0}

    # Increase the number of curveball steps until the null has moved at
    # least E edges (the draft's anti-degeneracy assertion).
    rewired = mask.copy()
    total_moves = 0
    for round_index in range(8):
        steps = n_swaps_factor * (2 ** round_index)
        candidate, moves = _curveball_rewire(mask, seed + round_index, steps)
        total_moves += moves
        hamming = int(np.logical_xor(mask, candidate).sum())
        if hamming >= n_edges:
            rewired = candidate
            break
        # Keep the best candidate seen so far.
        if hamming > int(np.logical_xor(mask, rewired).sum()):
            rewired = candidate
    else:
        rewired = candidate  # last candidate; assertion below reports failure

    _validate_bipartite_degrees(mask, rewired)
    hamming = int(np.logical_xor(mask, rewired).sum())
    if hamming < n_edges:
        raise MaskError(
            f"rewiring did not move enough edges: Hamming={hamming}, E={n_edges}; "
            "increase n_swaps_factor or inspect the mask"
        )
    info = {
        "n_edges": int(rewired.sum()),
        "n_moves": int(total_moves),
        "n_steps_factor": int(n_swaps_factor),
        "hamming": hamming,
        "hamming_fraction": float(hamming / mask.size),
    }
    return rewired, info


def directed_degree_preserving_rewire(
    adjacency: np.ndarray,
    seed: int,
    n_swaps_factor: int = 10,
    max_attempts_factor: int = 200,
) -> np.ndarray:
    """Directed edge swaps preserving in/out degrees, no self-loops/multiedges."""
    adjacency = np.asarray(adjacency, dtype=bool)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square")
    edge_set = {tuple(map(int, edge)) for edge in np.argwhere(adjacency)}
    edge_list = list(edge_set)
    n_edges = len(edge_list)
    if n_edges < 2:
        return adjacency.copy()

    rng = np.random.default_rng(int(seed))
    target_swaps = int(n_swaps_factor * n_edges)
    max_attempts = int(max_attempts_factor * n_edges)
    successful = 0

    for _ in range(max_attempts):
        if successful >= target_swaps:
            break
        i, j = rng.choice(len(edge_list), size=2, replace=False)
        a, b = edge_list[i]
        c, d = edge_list[j]
        if a == d or c == b or a == c or b == d:
            continue
        new1, new2 = (a, d), (c, b)
        if new1 in edge_set or new2 in edge_set:
            continue
        edge_set.remove((a, b))
        edge_set.remove((c, d))
        edge_set.add(new1)
        edge_set.add(new2)
        edge_list[i] = new1
        edge_list[j] = new2
        successful += 1

    rewired = np.zeros_like(adjacency, dtype=bool)
    for a, b in edge_set:
        rewired[a, b] = True
    if not np.array_equal(adjacency.sum(axis=0), rewired.sum(axis=0)):
        raise MaskError("directed in-degree sequence changed")
    if not np.array_equal(adjacency.sum(axis=1), rewired.sum(axis=1)):
        raise MaskError("directed out-degree sequence changed")
    return rewired


def directed_havel_hakimi_null(adjacency: np.ndarray, seed: int = 0) -> np.ndarray:
    """Fast simple directed null with the same in/out degree sequences."""
    adjacency = np.asarray(adjacency, dtype=bool)
    in_degrees = adjacency.sum(axis=0).astype(int).tolist()
    out_degrees = adjacency.sum(axis=1).astype(int).tolist()
    try:
        graph = nx.directed_havel_hakimi_graph(in_degrees, out_degrees)
    except Exception:
        return directed_degree_preserving_rewire(adjacency, seed=seed)
    null = nx.to_numpy_array(graph, nodelist=sorted(graph.nodes())).astype(bool)
    if not np.array_equal(adjacency.sum(axis=0), null.sum(axis=0)):
        raise MaskError("Havel-Hakimi null changed in-degree sequence")
    if not np.array_equal(adjacency.sum(axis=1), null.sum(axis=1)):
        raise MaskError("Havel-Hakimi null changed out-degree sequence")
    return null


def degree_distribution_table(
    bio_mask: np.ndarray,
    deg_mask: np.ndarray,
    er_mask: np.ndarray,
) -> list[dict]:
    """Rows for the draft's Figure 1C (out-degree vs rank)."""
    bio = sorted(np.asarray(bio_mask).sum(axis=1).astype(int), reverse=True)
    deg = sorted(np.asarray(deg_mask).sum(axis=1).astype(int), reverse=True)
    er = sorted(np.asarray(er_mask).sum(axis=1).astype(int), reverse=True)
    rows = []
    for rank in range(len(bio)):
        rows.append(
            {
                "rank": rank + 1,
                "bio_out_degree": bio[rank],
                "deg_rewired_out_degree": deg[rank],
                "er_random_out_degree": er[rank],
            }
        )
    return rows
