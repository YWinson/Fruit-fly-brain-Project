"""Cell-type clustering and granularity probes.

The draft specifies three granularities:
* fine   K_out=64, K_in=32  (terminal-type clusters)
* medium K_out=32, K_in=16  (functional-group / hemilineage-like clusters)
* coarse K_out=2,  K_in=2   (pure E/I negative boundary)

To keep the decoder layer at ``d_out x d_in = 64 x 32`` for every arm, the
cluster matrix is expanded by repeating rows/columns (see ``expand_cluster``).
This is an explicit operationalisation of the draft's "align the fly graph to
64 x 32"; it keeps the architecture and parameter budget identical across
granularity probes.

Only the sensorimotor / mushroom-body subset is used by default (the draft's
scope); the optic-lobe types are excluded because the draft explicitly moves
away from flyvis's periodic optic-lobe tiling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

SENSORIMOTOR_SUPERCLASSES = (
    "central",
    "sensory",
    "ascending",
    "descending",
    "motor",
    "sensory_ascending",
    "endocrine",
)

NEUROTRANSMITTERS = (
    "acetylcholine",
    "glutamate",
    "gaba",
    "dopamine",
    "serotonin",
    "octopamine",
)


@dataclass(frozen=True)
class Granularity:
    name: str
    k_out: int
    k_in: int

    @property
    def expand_out(self) -> int:
        if 64 % self.k_out:
            raise ValueError(f"k_out={self.k_out} does not divide 64")
        return 64 // self.k_out

    @property
    def expand_in(self) -> int:
        if 32 % self.k_in:
            raise ValueError(f"k_in={self.k_in} does not divide 32")
        return 32 // self.k_in


GRANULARITIES = {
    "fine": Granularity("fine", 64, 32),
    "medium": Granularity("medium", 32, 16),
    "coarse": Granularity("coarse", 2, 2),
}


@dataclass
class GranularityResult:
    name: str
    k_out: int
    k_in: int
    selected_type_indices: np.ndarray
    sub_meta: pd.DataFrame
    out_labels: np.ndarray
    in_labels: np.ndarray
    W_syn_cluster: np.ndarray
    W_exc_cluster: np.ndarray
    W_inh_cluster: np.ndarray
    W_edge_cluster: np.ndarray
    W_syn_expanded: np.ndarray
    W_exc_expanded: np.ndarray
    W_inh_expanded: np.ndarray
    W_syn_square: np.ndarray
    W_exc_square: np.ndarray
    W_inh_square: np.ndarray
    expand_out: int
    expand_in: int
    stats: dict


def select_type_indices(
    tc,
    superclasses: tuple[str, ...] = SENSORIMOTOR_SUPERCLASSES,
    min_neurons: int = 1,
) -> np.ndarray:
    """Indices of cell types in the sensorimotor/MB subset."""
    meta = tc.type_meta
    mask = meta["super_class"].isin(superclasses) & (meta["n_neurons"] >= min_neurons)
    indices = np.flatnonzero(mask.to_numpy())
    if len(indices) < 4:
        raise ValueError(f"only {len(indices)} types selected; check superclass filter")
    return indices


def _superclass_onehot(meta: pd.DataFrame) -> sp.csr_matrix:
    classes = sorted(meta["super_class"].fillna("").astype(str).unique())
    class_to_idx = {name: index for index, name in enumerate(classes)}
    rows = np.arange(len(meta))
    cols = meta["super_class"].fillna("").astype(str).map(class_to_idx).to_numpy()
    return sp.csr_matrix(
        (np.ones(len(meta)), (rows, cols)), shape=(len(meta), len(classes))
    )


def build_type_features(
    tc,
    selected_indices: np.ndarray,
) -> np.ndarray:
    """Connectivity-profile + neurotransmitter features for type clustering."""
    sub_meta = tc.type_meta.iloc[selected_indices].reset_index(drop=True)
    W = tc.W_syn[selected_indices][:, selected_indices].astype(np.float64)
    S = _superclass_onehot(sub_meta)

    out_profile = np.asarray((W @ S).todense(), dtype=np.float64)
    in_profile = np.asarray((W.T @ S).todense(), dtype=np.float64)

    def _normalise(matrix: np.ndarray) -> np.ndarray:
        row_sum = matrix.sum(axis=1, keepdims=True)
        row_sum[row_sum <= 0] = 1.0
        return matrix / row_sum

    out_profile = _normalise(out_profile)
    in_profile = _normalise(in_profile)

    nt = sub_meta["top_nt"].fillna("unknown").astype(str).to_numpy()
    nt_features = np.zeros((len(sub_meta), len(NEUROTRANSMITTERS) + 1))
    for index, value in enumerate(nt):
        if value in NEUROTRANSMITTERS:
            nt_features[index, NEUROTRANSMITTERS.index(value)] = 1.0
        else:
            nt_features[index, -1] = 1.0

    size_features = np.column_stack(
        [
            np.log1p(sub_meta["n_neurons"].to_numpy(dtype=np.float64)),
            np.log1p(sub_meta["total_synapses"].to_numpy(dtype=np.float64)),
        ]
    )
    features = np.hstack([out_profile, in_profile, nt_features, size_features])
    return StandardScaler().fit_transform(features)


def ei_labels(sub_meta: pd.DataFrame) -> np.ndarray:
    """Pure E/I grouping for the coarse negative boundary.

    Following the draft's simplification: acetylcholine and glutamate are
    excitatory, GABA is inhibitory; other neurotransmitters default to
    excitatory. This is a coarse boundary, not a claim about receptor-level
    sign for every neuron.
    """
    nt = sub_meta["top_nt"].fillna("").astype(str).str.lower()
    labels = np.zeros(len(sub_meta), dtype=np.int32)
    labels[nt.str.contains("gaba")] = 1
    return labels


def cluster_labels(features: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Ward agglomerative clustering into exactly ``k`` groups."""
    if k < 1:
        raise ValueError("k must be >= 1")
    if k == 1:
        return np.zeros(len(features), dtype=np.int32)
    model = AgglomerativeClustering(n_clusters=k, linkage="ward")
    return model.fit_predict(features).astype(np.int32)


def _onehot_aggregation(labels: np.ndarray, k: int, axis: str) -> sp.csr_matrix:
    n = len(labels)
    rows = labels if axis == "out" else np.arange(n)
    cols = np.arange(n) if axis == "out" else labels
    shape = (k, n) if axis == "out" else (n, k)
    return sp.csr_matrix((np.ones(n), (rows, cols)), shape=shape)


def _aggregate(
    W: sp.csr_matrix,
    out_labels: np.ndarray,
    in_labels: np.ndarray,
    k_out: int,
    k_in: int,
) -> np.ndarray:
    A_out = _onehot_aggregation(out_labels, k_out, "out")
    A_in = _onehot_aggregation(in_labels, k_in, "in")
    return np.asarray((A_out @ W @ A_in).todense(), dtype=np.float64)


def expand_cluster(matrix: np.ndarray, expand_out: int, expand_in: int) -> np.ndarray:
    """Repeat rows/columns to reach the fixed 64 x 32 layer."""
    return np.repeat(np.repeat(matrix, expand_out, axis=0), expand_in, axis=1)


def build_granularity(
    tc,
    granularity: Granularity | str,
    superclasses: tuple[str, ...] = SENSORIMOTOR_SUPERCLASSES,
    min_neurons: int = 1,
    seed: int = 0,
) -> GranularityResult:
    """Build cluster-level and 64x32-expanded connectivity for one granularity."""
    if isinstance(granularity, str):
        granularity = GRANULARITIES[granularity]

    selected = select_type_indices(tc, superclasses=superclasses, min_neurons=min_neurons)
    sub_meta = tc.type_meta.iloc[selected].reset_index(drop=True)

    if granularity.name == "coarse":
        out_labels = ei_labels(sub_meta)
        in_labels = ei_labels(sub_meta)
    else:
        features = build_type_features(tc, selected)
        out_labels = cluster_labels(features, granularity.k_out, seed=seed)
        in_labels = cluster_labels(features, granularity.k_in, seed=seed)

    W_sub = tc.W_syn[selected][:, selected].astype(np.float64)
    We_sub = tc.W_exc[selected][:, selected].astype(np.float64)
    Wi_sub = tc.W_inh[selected][:, selected].astype(np.float64)
    Wd_sub = tc.W_edge[selected][:, selected].astype(np.float64)

    W_cluster = _aggregate(W_sub, out_labels, in_labels, granularity.k_out, granularity.k_in)
    We_cluster = _aggregate(
        We_sub, out_labels, in_labels, granularity.k_out, granularity.k_in
    )
    Wi_cluster = _aggregate(
        Wi_sub, out_labels, in_labels, granularity.k_out, granularity.k_in
    )
    Wd_cluster = _aggregate(
        Wd_sub, out_labels, in_labels, granularity.k_out, granularity.k_in
    )

    # Square cluster graph on the output clusters (for motif/δ_topo analysis).
    W_square = _aggregate(
        W_sub, out_labels, out_labels, granularity.k_out, granularity.k_out
    )
    We_square = _aggregate(
        We_sub, out_labels, out_labels, granularity.k_out, granularity.k_out
    )
    Wi_square = _aggregate(
        Wi_sub, out_labels, out_labels, granularity.k_out, granularity.k_out
    )

    W_expanded = expand_cluster(
        W_cluster, granularity.expand_out, granularity.expand_in
    )
    We_expanded = expand_cluster(
        We_cluster, granularity.expand_out, granularity.expand_in
    )
    Wi_expanded = expand_cluster(
        Wi_cluster, granularity.expand_out, granularity.expand_in
    )

    stats = {
        "granularity": granularity.name,
        "k_out": granularity.k_out,
        "k_in": granularity.k_in,
        "expand_out": granularity.expand_out,
        "expand_in": granularity.expand_in,
        "n_types_selected": int(len(selected)),
        "n_neurons_selected": int(sub_meta["n_neurons"].sum()),
        "n_nonzero_cluster_pairs": int((W_cluster > 0).sum()),
        "n_nonzero_square_pairs": int((W_square > 0).sum()),
        "n_nonzero_expanded_pairs": int((W_expanded > 0).sum()),
        "cluster_density": float((W_cluster > 0).mean()),
        "expanded_density": float((W_expanded > 0).mean()),
        "total_synapses_cluster": float(W_cluster.sum()),
        "excitatory_synapses_cluster": float(We_cluster.sum()),
        "inhibitory_synapses_cluster": float(Wi_cluster.sum()),
        "out_cluster_sizes": np.bincount(out_labels, minlength=granularity.k_out).tolist(),
        "in_cluster_sizes": np.bincount(in_labels, minlength=granularity.k_in).tolist(),
    }
    return GranularityResult(
        name=granularity.name,
        k_out=granularity.k_out,
        k_in=granularity.k_in,
        selected_type_indices=selected,
        sub_meta=sub_meta.assign(out_cluster=out_labels, in_cluster=in_labels),
        out_labels=out_labels,
        in_labels=in_labels,
        W_syn_cluster=W_cluster,
        W_exc_cluster=We_cluster,
        W_inh_cluster=Wi_cluster,
        W_edge_cluster=Wd_cluster,
        W_syn_expanded=W_expanded,
        W_exc_expanded=We_expanded,
        W_inh_expanded=Wi_expanded,
        W_syn_square=W_square,
        W_exc_square=We_square,
        W_inh_square=Wi_square,
        expand_out=granularity.expand_out,
        expand_in=granularity.expand_in,
        stats=stats,
    )


def save_granularity(result: GranularityResult, out_dir: str | Path) -> Path:
    """Save cluster assignments and matrices for audit/reproduction."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.sub_meta.to_csv(
        out_dir / f"clusters_{result.name}.csv", index=False, encoding="utf-8"
    )
    np.savez_compressed(
        out_dir / f"granularity_{result.name}.npz",
        W_syn_cluster=result.W_syn_cluster,
        W_exc_cluster=result.W_exc_cluster,
        W_inh_cluster=result.W_inh_cluster,
        W_edge_cluster=result.W_edge_cluster,
        W_syn_expanded=result.W_syn_expanded,
        W_exc_expanded=result.W_exc_expanded,
        W_inh_expanded=result.W_inh_expanded,
        W_syn_square=result.W_syn_square,
        W_exc_square=result.W_exc_square,
        W_inh_square=result.W_inh_square,
        out_labels=result.out_labels,
        in_labels=result.in_labels,
    )
    return out_dir
