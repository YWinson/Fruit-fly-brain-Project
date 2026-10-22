"""Motif-based δ_topo calibration (draft figure 1A)."""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np

from .masks import directed_degree_preserving_rewire, directed_havel_hakimi_null


def _binary_adjacency(adjacency: np.ndarray, exclude_self_loops: bool = True) -> np.ndarray:
    adjacency = np.asarray(adjacency) > 0
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    if exclude_self_loops:
        adjacency = adjacency.copy()
        np.fill_diagonal(adjacency, False)
    return adjacency.astype(np.int64)


def ffl_count(adjacency: np.ndarray) -> int:
    """Count directed feed-forward loops a->b, a->c, b->c."""
    A = _binary_adjacency(adjacency)
    return int(np.einsum("ab,ac,bc->", A, A, A))


def bifan_count(adjacency: np.ndarray) -> int:
    """Count bi-fans: two sources and two targets, fully connected."""
    A = _binary_adjacency(adjacency)
    common_sources = A.T @ A  # [c, d] = number of sources connected to both
    total = 0
    n = common_sources.shape[0]
    for c in range(n):
        for d in range(c + 1, n):
            value = int(common_sources[c, d])
            total += value * (value - 1) // 2
    return int(total)


def motif_counts(adjacency: np.ndarray) -> dict:
    return {
        "ffl": ffl_count(adjacency),
        "bifan": bifan_count(adjacency),
    }


def _null_graph(adjacency: np.ndarray, seed: int, method: str) -> np.ndarray:
    if method == "edge_swap":
        # NetworkX's directed triple-edge swap preserves in/out degrees and
        # avoids parallel edges; it is the randomized null the draft asks for.
        graph = nx.from_numpy_array(
            np.asarray(adjacency, dtype=np.int8), create_using=nx.DiGraph
        )
        n_edges = int(np.asarray(adjacency).sum())
        # 2xE triple-edge swaps are enough to randomise the null; 10xE was
        # needlessly slow for a 2,583-edge cluster graph.
        nswap = max(1, int(2 * n_edges))
        max_tries = max(1000, int(10 * nswap))
        try:
            graph = nx.directed_edge_swap(
                graph, nswap=nswap, max_tries=max_tries, seed=seed
            )
            null = nx.to_numpy_array(
                graph, nodelist=sorted(graph.nodes())
            ).astype(bool)
        except Exception:
            null = directed_degree_preserving_rewire(adjacency, seed=seed)
    elif method == "havel_hakimi":
        # Deterministic generator; useful as a sanity check but its null
        # distribution has zero variance.
        null = directed_havel_hakimi_null(adjacency, seed=seed)
    elif method == "swap":
        null = directed_degree_preserving_rewire(adjacency, seed=seed)
    else:
        raise ValueError(f"unknown null method: {method!r}")
    null = np.asarray(null, dtype=bool)
    np.fill_diagonal(null, False)
    return null


def _null_motif_counts(A: np.ndarray, seed: int, method: str):
    try:
        null = _null_graph(A, seed=seed, method=method)
    except Exception:
        if method == "swap":
            return None
        try:
            null = _null_graph(A, seed=seed, method="swap")
        except Exception:
            return None
    counts = motif_counts(null)
    return counts["ffl"], counts["bifan"]


def calibrate_motifs(
    adjacency: np.ndarray,
    n_samples: int = 1000,
    seed: int = 0,
    method: str = "edge_swap",
    n_jobs: int = 1,
) -> dict:
    """Compare observed FFL/bi-fan counts with degree-preserving nulls.

    Returns observed counts, null mean/std, the 99th percentile (the draft's
    δ_topo), Z-scores, empirical p-values, and the raw null arrays.
    """
    A = _binary_adjacency(adjacency)
    observed = motif_counts(A)

    if n_jobs in (None, 0, 1):
        results = [
            _null_motif_counts(A, seed=seed + index, method=method)
            for index in range(int(n_samples))
        ]
    else:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs)(
            delayed(_null_motif_counts)(A, seed + index, method)
            for index in range(int(n_samples))
        )
    results = [value for value in results if value is not None]
    if not results:
        raise RuntimeError("no valid null graphs generated")
    null_ffl = np.asarray([value[0] for value in results], dtype=np.int64)
    null_bifan = np.asarray([value[1] for value in results], dtype=np.int64)
    used = len(results)

    def _summary(values: np.ndarray, obs: int) -> dict:
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        q99 = float(np.percentile(values, 99))
        z = (obs - mean) / std if std > 0 else float("nan")
        p = float((values >= obs).mean())
        return {
            "observed": int(obs),
            "null_mean": mean,
            "null_std": std,
            "null_q99_delta_topo": q99,
            "z": z,
            "p_empirical": p,
        }

    return {
        "n_requested": int(n_samples),
        "n_valid": int(used),
        "method": method,
        "ffl": _summary(null_ffl, observed["ffl"]),
        "bifan": _summary(null_bifan, observed["bifan"]),
        "null_ffl": null_ffl.tolist(),
        "null_bifan": null_bifan.tolist(),
    }


def save_calibration(result: dict, out_dir: str | Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "motif_calibration.json"
    payload = {key: value for key, value in result.items() if not key.startswith("null_")}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(
        out_dir / "motif_null_distributions.npz",
        ffl=np.asarray(result["null_ffl"]),
        bifan=np.asarray(result["null_bifan"]),
    )
    return path


def plot_motif_null(result: dict, out_path: str | Path) -> Path:
    """Histogram of null FFL counts with the observed count and 99% line."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for axis, key, label in zip(
        axes, ("ffl", "bifan"), ("FFL", "bi-fan")
    ):
        summary = result[key]
        values = np.asarray(result[f"null_{key}"])
        axis.hist(values, bins=40, color="#9aa7b1", edgecolor="white", linewidth=0.4)
        axis.axvline(summary["observed"], color="#c0392b", lw=2, label="observed")
        axis.axvline(
            summary["null_q99_delta_topo"],
            color="#2c3e50",
            lw=1.5,
            ls="--",
            label="null 99% (δ_topo)",
        )
        axis.set_title(f"{label}: Z={summary['z']:.2f}, p={summary['p_empirical']:.3f}")
        axis.set_xlabel("motif count")
        axis.set_ylabel("null graphs")
        axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path
