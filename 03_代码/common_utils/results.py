"""IF-2 result table and per-trial prediction export.

The fixed header comes from the task book's IF-2 contract:
``arm/model/subject/session/seed/balanced_acc/latency_ms/memory_MB/note`` plus
the ablation fields. Extra columns are allowed, but the required ones are
always present.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .paths import ensure_dir

RESULT_COLUMNS = [
    "arm",
    "model",
    "subject",
    "session",
    "seed",
    "balanced_acc",
    "latency_ms",
    "memory_MB",
    "note",
    "ablation_target",
    "mask_id",
    "activation_map_path",
]

PREDICTION_COLUMNS = [
    "arm",
    "model",
    "subject",
    "session",
    "run",
    "trial",
    "epoch_id",
    "seed",
    "fold_id",
    "y_true",
    "y_pred",
    "y_score",
    "label_true",
    "label_pred",
]


def ensure_result_columns(results: pd.DataFrame) -> pd.DataFrame:
    """Add missing IF-2 columns without dropping extra columns."""
    results = results.copy()
    for column in RESULT_COLUMNS:
        if column not in results.columns:
            results[column] = ""
    return results


def ensure_prediction_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    """Add missing prediction columns without dropping extra columns."""
    predictions = predictions.copy()
    for column in PREDICTION_COLUMNS:
        if column not in predictions.columns:
            predictions[column] = ""
    return predictions


def validate_results(results: pd.DataFrame) -> pd.DataFrame:
    """Validate the IF-2 result table before writing it."""
    missing = [column for column in RESULT_COLUMNS if column not in results.columns]
    if missing:
        raise ValueError(f"results table is missing IF-2 columns: {missing}")

    scores = pd.to_numeric(results["balanced_acc"], errors="coerce")
    finite = scores.dropna()
    if ((finite < 0) | (finite > 1)).any():
        raise ValueError("balanced_acc must be in [0, 1] (or NaN)")

    if results["seed"].isnull().any():
        raise ValueError("seed must not be null")

    return results


def write_results(
    results: pd.DataFrame,
    out_dir: str | Path,
    run_id: str,
) -> tuple[Path, Path]:
    """Write ``results_<run_id>.csv`` and ``.json``; return both paths."""
    out_dir = ensure_dir(out_dir)
    results = ensure_result_columns(results)
    validate_results(results)

    csv_path = out_dir / f"results_{run_id}.csv"
    json_path = out_dir / f"results_{run_id}.json"
    results.to_csv(csv_path, index=False, encoding="utf-8-sig")
    results.to_json(json_path, orient="records", force_ascii=False, indent=2)
    return csv_path, json_path


def write_predictions(
    predictions: pd.DataFrame,
    out_dir: str | Path,
    run_id: str,
) -> Path:
    """Write per-trial predictions, the raw material for B's statistics."""
    out_dir = ensure_dir(out_dir)
    predictions = ensure_prediction_columns(predictions)
    path = out_dir / f"predictions_{run_id}.csv"
    predictions.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def aggregate_results(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate by arm/model/seed; prefer the subject-level ``session='all'`` rows."""
    overall = results
    if "session" in results.columns and (results["session"].astype(str) == "all").any():
        overall = results[results["session"].astype(str) == "all"]
    return (
        overall.groupby(["arm", "model", "seed"], dropna=False)["balanced_acc"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
