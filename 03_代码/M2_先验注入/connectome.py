"""Build a cell-type-level connectivity matrix from FlyWire v783.

Data sources
------------
* ``Connectivity_783.parquet``: FlyWire v783 proofread neuron-pair edge list
  (Shiu et al. model repository / FlyWire.ai). Columns used:
  ``Presynaptic_ID``, ``Postsynaptic_ID``, ``Connectivity``, ``Excitatory``.
* ``Supplemental_file1_neuron_annotations.tsv``: Schlegel et al. (2024)
  FlyWire annotations (cell types, superclasses, neurotransmitters, ...).

The output is a sparse cell-type x cell-type matrix of synapse counts, plus
excitatory/inhibitory splits and the number of neuron pairs per type pair.
Edges are filtered with the draft's ``Connectivity >= 5`` threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from pyarrow import parquet as pq

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from common_utils.paths import FLY_PROCESSED_DIR, FLY_RAW_DIR, ensure_dir

ANNOTATION_COLUMNS = [
    "root_id",
    "cell_type",
    "super_class",
    "cell_class",
    "cell_sub_class",
    "supertype",
    "top_nt",
    "side",
    "nerve",
]

CONNECTIVITY_COLUMNS = [
    "Presynaptic_ID",
    "Postsynaptic_ID",
    "Connectivity",
    "Excitatory",
]


def _mode_or_empty(values: pd.Series) -> str:
    values = values.dropna()
    if values.empty:
        return ""
    mode = values.mode()
    return str(mode.iloc[0]) if len(mode) else str(values.iloc[0])


@dataclass
class TypeConnectome:
    """Sparse cell-type connectivity matrices and their metadata."""

    W_syn: sp.csr_matrix
    W_exc: sp.csr_matrix
    W_inh: sp.csr_matrix
    W_edge: sp.csr_matrix
    type_meta: pd.DataFrame
    type_ids: list[str]
    summary: dict


def load_annotations(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load annotations and add an integer ``type_idx`` column.

    Returns ``(neurons, type_meta, type_ids)``. ``neurons`` has one row per
    annotated root id; ``type_meta`` has one row per cell type.
    """
    path = Path(path)
    df = pd.read_csv(
        path,
        sep="\t",
        usecols=ANNOTATION_COLUMNS,
        low_memory=False,
    )
    df = df.dropna(subset=["root_id", "cell_type"]).copy()
    df["root_id"] = df["root_id"].astype(np.int64)
    df["cell_type"] = df["cell_type"].astype(str)
    df = df.drop_duplicates("root_id", keep="first").reset_index(drop=True)

    type_ids = sorted(df["cell_type"].unique().tolist())
    type_to_idx = {name: index for index, name in enumerate(type_ids)}
    df["type_idx"] = df["cell_type"].map(type_to_idx).astype(np.int32)

    meta = (
        df.groupby("type_idx", sort=True)
        .agg(
            type_id=("cell_type", "first"),
            n_neurons=("root_id", "size"),
            super_class=("super_class", _mode_or_empty),
            cell_class=("cell_class", _mode_or_empty),
            cell_sub_class=("cell_sub_class", _mode_or_empty),
            supertype=("supertype", _mode_or_empty),
            top_nt=("top_nt", _mode_or_empty),
            side=("side", _mode_or_empty),
            nerve=("nerve", _mode_or_empty),
        )
        .reset_index()
    )
    if len(meta) != len(type_ids):
        raise RuntimeError(
            f"type metadata has {len(meta)} rows but {len(type_ids)} type ids"
        )
    return df, meta, type_ids


def build_type_connectome(
    neurons: pd.DataFrame,
    type_meta: pd.DataFrame,
    type_ids: list[str],
    connectivity_path: str | Path,
    min_synapses: int = 5,
    batch_size: int = 2_000_000,
) -> TypeConnectome:
    """Aggregate neuron-pair edges into a cell-type x cell-type matrix."""
    connectivity_path = Path(connectivity_path)
    if not connectivity_path.exists():
        raise FileNotFoundError(
            f"{connectivity_path} is missing; download the FlyWire v783 "
            "Connectivity_783.parquet first (see M2 README)"
        )

    roots = neurons["root_id"].to_numpy(dtype=np.int64)
    order = np.argsort(roots)
    sorted_roots = roots[order]
    sorted_type = neurons["type_idx"].to_numpy(dtype=np.int32)[order]
    n_types = len(type_ids)

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    syn_weights: list[np.ndarray] = []
    exc_weights: list[np.ndarray] = []
    inh_weights: list[np.ndarray] = []
    edge_weights: list[np.ndarray] = []

    parquet = pq.ParquetFile(connectivity_path)
    n_raw_edges = 0
    n_kept_edges = 0
    n_unmapped = 0

    for batch in parquet.iter_batches(
        batch_size=batch_size, columns=CONNECTIVITY_COLUMNS
    ):
        frame = batch.to_pandas()
        n_raw_edges += len(frame)
        frame = frame[frame["Connectivity"] >= min_synapses]
        if frame.empty:
            continue

        pre_ids = frame["Presynaptic_ID"].to_numpy(dtype=np.int64)
        post_ids = frame["Postsynaptic_ID"].to_numpy(dtype=np.int64)
        connectivity = frame["Connectivity"].to_numpy(dtype=np.float64)
        excitatory = frame["Excitatory"].to_numpy(dtype=np.float64)

        pre_pos = np.clip(np.searchsorted(sorted_roots, pre_ids), 0, len(sorted_roots) - 1)
        post_pos = np.clip(
            np.searchsorted(sorted_roots, post_ids), 0, len(sorted_roots) - 1
        )
        valid = (sorted_roots[pre_pos] == pre_ids) & (
            sorted_roots[post_pos] == post_ids
        )
        n_unmapped += int((~valid).sum())
        if not valid.any():
            continue

        pre_type = sorted_type[pre_pos[valid]]
        post_type = sorted_type[post_pos[valid]]
        conn = connectivity[valid]
        exc = excitatory[valid]
        n_kept_edges += int(valid.sum())

        rows.append(pre_type.astype(np.int32))
        cols.append(post_type.astype(np.int32))
        syn_weights.append(conn)
        exc_weights.append(np.where(exc > 0, conn, 0.0))
        inh_weights.append(np.where(exc < 0, conn, 0.0))
        edge_weights.append(np.ones(len(conn), dtype=np.float64))

    if not rows:
        raise RuntimeError("no edges survived filtering/mapping")

    row = np.concatenate(rows)
    col = np.concatenate(cols)
    shape = (n_types, n_types)

    def _matrix(weights: list[np.ndarray]) -> sp.csr_matrix:
        return sp.coo_matrix(
            (np.concatenate(weights), (row, col)), shape=shape
        ).tocsr()

    W_syn = _matrix(syn_weights)
    W_exc = _matrix(exc_weights)
    W_inh = _matrix(inh_weights)
    W_edge = _matrix(edge_weights)

    total_in = np.asarray(W_syn.sum(axis=0)).ravel()
    total_out = np.asarray(W_syn.sum(axis=1)).ravel()
    degree = np.asarray((W_syn > 0).sum(axis=0)).ravel() + np.asarray(
        (W_syn > 0).sum(axis=1)
    ).ravel()
    meta = type_meta.copy()
    meta["total_in_synapses"] = total_in
    meta["total_out_synapses"] = total_out
    meta["total_synapses"] = total_in + total_out
    meta["total_degree"] = degree

    top = (
        meta.sort_values("total_synapses", ascending=False)
        .head(15)[["type_id", "n_neurons", "total_synapses"]]
        .to_dict(orient="records")
    )
    summary = {
        "n_annotated_neurons": int(len(neurons)),
        "n_types": int(n_types),
        "n_raw_edges": int(n_raw_edges),
        "n_kept_edges": int(n_kept_edges),
        "n_unmapped_edges": int(n_unmapped),
        "min_synapses": int(min_synapses),
        "total_synapses": float(W_syn.sum()),
        "excitatory_synapses": float(W_exc.sum()),
        "inhibitory_synapses": float(W_inh.sum()),
        "n_nonzero_type_pairs": int(W_syn.nnz),
        "density_type_matrix": float(W_syn.nnz / (n_types * n_types)),
        "top_types": top,
    }
    return TypeConnectome(
        W_syn=W_syn,
        W_exc=W_exc,
        W_inh=W_inh,
        W_edge=W_edge,
        type_meta=meta,
        type_ids=type_ids,
        summary=summary,
    )


def save_type_connectome(tc: TypeConnectome, out_dir: str | Path) -> Path:
    """Save sparse matrices and metadata under ``out_dir``."""
    out_dir = ensure_dir(out_dir)
    sp.save_npz(out_dir / "W_syn.npz", tc.W_syn)
    sp.save_npz(out_dir / "W_exc.npz", tc.W_exc)
    sp.save_npz(out_dir / "W_inh.npz", tc.W_inh)
    sp.save_npz(out_dir / "W_edge.npz", tc.W_edge)
    tc.type_meta.to_csv(out_dir / "type_meta.csv", index=False, encoding="utf-8")
    pd.DataFrame(
        {"type_idx": np.arange(len(tc.type_ids)), "cell_type": tc.type_ids}
    ).to_csv(out_dir / "type_ids.csv", index=False, encoding="utf-8")
    (out_dir / "summary.json").write_text(
        json.dumps(tc.summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return out_dir


def load_type_connectome(in_dir: str | Path = FLY_PROCESSED_DIR) -> TypeConnectome:
    """Load a saved :class:`TypeConnectome`."""
    in_dir = Path(in_dir)
    meta = pd.read_csv(in_dir / "type_meta.csv")
    type_ids = pd.read_csv(in_dir / "type_ids.csv")["cell_type"].astype(str).tolist()
    summary = json.loads((in_dir / "summary.json").read_text(encoding="utf-8"))
    return TypeConnectome(
        W_syn=sp.load_npz(in_dir / "W_syn.npz"),
        W_exc=sp.load_npz(in_dir / "W_exc.npz"),
        W_inh=sp.load_npz(in_dir / "W_inh.npz"),
        W_edge=sp.load_npz(in_dir / "W_edge.npz"),
        type_meta=meta,
        type_ids=type_ids,
        summary=summary,
    )


def build_from_defaults(
    raw_dir: str | Path = FLY_RAW_DIR,
    out_dir: str | Path = FLY_PROCESSED_DIR,
    min_synapses: int = 5,
    force: bool = False,
) -> TypeConnectome:
    """Convenience wrapper used by the M2 CLI/tests."""
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    if (out_dir / "W_syn.npz").exists() and not force:
        return load_type_connectome(out_dir)

    annotations_path = raw_dir / "Supplemental_file1_neuron_annotations.tsv"
    connectivity_path = raw_dir / "Connectivity_783.parquet"
    neurons, meta, type_ids = load_annotations(annotations_path)
    tc = build_type_connectome(
        neurons,
        meta,
        type_ids,
        connectivity_path,
        min_synapses=min_synapses,
    )
    save_type_connectome(tc, out_dir)
    return tc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build FlyWire v783 cell-type connectivity matrices"
    )
    parser.add_argument("--raw-dir", default=str(FLY_RAW_DIR))
    parser.add_argument("--out-dir", default=str(FLY_PROCESSED_DIR))
    parser.add_argument("--min-synapses", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    tc = build_from_defaults(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        min_synapses=args.min_synapses,
        force=args.force,
    )
    print(json.dumps(tc.summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
