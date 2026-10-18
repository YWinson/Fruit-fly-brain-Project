"""LOSO evaluation for M1.

Each fold:
1. runs the split-level leakage checks again;
2. builds a fresh pipeline (preprocessing + arm-specific mask + classifier);
3. fits only on the training subjects;
4. predicts the held-out subject's sessions;
5. writes per-session and overall balanced accuracy plus per-trial predictions.
"""

from __future__ import annotations

import json
import time
import tracemalloc
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from common_utils.leakage_checks import (
    CanaryResult,
    label_permutation_canary,
    run_fold_checks,
)
from common_utils.results import (
    aggregate_results,
    ensure_prediction_columns,
    ensure_result_columns,
)
from common_utils.seeding import derive_seed, set_global_seed
from common_utils.splits import Fold, make_loso_splits

from .arms import arm_mask_id, arm_model_name, build_arm


def _safe_balanced_accuracy(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return float("nan")
    return float(balanced_accuracy_score(y_true, y_pred))


def _names_for(values, class_names: list[str]) -> list[str]:
    names = []
    for value in np.asarray(values):
        index = int(value)
        names.append(class_names[index] if 0 <= index < len(class_names) else str(value))
    return names


def evaluate_fold(
    dataset,
    fold: Fold,
    arm: str,
    seed: int,
    config: dict,
) -> tuple[list[dict], list[dict]]:
    """Fit one arm on one fold and return result rows and prediction rows."""
    run_fold_checks(dataset.metadata, fold)

    train_idx = np.asarray(fold.train_idx, dtype=int)
    test_idx = np.asarray(fold.test_idx, dtype=int)
    X_train, y_train = dataset.X[train_idx], dataset.y[train_idx]
    X_test, y_test = dataset.X[test_idx], dataset.y[test_idx]

    # Derive a fold-specific seed so a rerun of the same fold is reproducible.
    set_global_seed(derive_seed(seed, fold.fold_id, arm))
    pipeline = build_arm(arm, config, seed=seed, fold_id=fold.fold_id)

    tracemalloc.start()
    fit_start = time.perf_counter()
    pipeline.fit(X_train, y_train)
    fit_time_s = time.perf_counter() - fit_start

    predict_start = time.perf_counter()
    y_pred = np.asarray(pipeline.predict(X_test))
    predict_time_s = time.perf_counter() - predict_start
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    if hasattr(pipeline, "predict_proba"):
        probabilities = np.asarray(pipeline.predict_proba(X_test))
        y_score = probabilities.max(axis=1)
    else:
        y_score = np.full(len(y_pred), np.nan)

    test_meta = dataset.metadata.iloc[test_idx].reset_index(drop=True)
    subject = str(fold.test_subjects[0])
    model_name = arm_model_name(arm)
    mask_id = arm_mask_id(arm, config, seed=seed, fold_id=fold.fold_id)
    latency_ms = (predict_time_s / max(1, len(y_test))) * 1000.0
    memory_mb = peak_bytes / (1024.0 ** 2)

    note = json.dumps(
        {
            "dataset": config.get("dataset", {}).get("name"),
            "fold_id": int(fold.fold_id),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "fit_time_s": round(float(fit_time_s), 6),
            "train_subjects": list(fold.train_subjects),
            "test_subjects": list(fold.test_subjects),
        },
        ensure_ascii=False,
    )

    rows: list[dict] = []
    session_values = test_meta["session_id"].astype(str).values
    for session in sorted(set(session_values.tolist())):
        mask = session_values == session
        rows.append(
            {
                "arm": arm,
                "model": model_name,
                "subject": subject,
                "session": session,
                "seed": int(seed),
                "balanced_acc": _safe_balanced_accuracy(y_test[mask], y_pred[mask]),
                "latency_ms": latency_ms,
                "memory_MB": memory_mb,
                "note": note,
                "ablation_target": "",
                "mask_id": mask_id,
                "activation_map_path": "",
            }
        )

    rows.append(
        {
            "arm": arm,
            "model": model_name,
            "subject": subject,
            "session": "all",
            "seed": int(seed),
            "balanced_acc": _safe_balanced_accuracy(y_test, y_pred),
            "latency_ms": latency_ms,
            "memory_MB": memory_mb,
            "note": note,
            "ablation_target": "",
            "mask_id": mask_id,
            "activation_map_path": "",
        }
    )

    prediction_rows: list[dict] = []
    label_true = _names_for(y_test, list(dataset.class_names or []))
    label_pred = _names_for(y_pred, list(dataset.class_names or []))
    for position in range(len(y_test)):
        prediction_rows.append(
            {
                "arm": arm,
                "model": model_name,
                "subject": str(test_meta.loc[position, "subject_id"]),
                "session": str(test_meta.loc[position, "session_id"]),
                "run": str(test_meta.loc[position, "run_id"]),
                "trial": str(test_meta.loc[position, "trial_id"]),
                "epoch_id": str(test_meta.loc[position, "epoch_id"]),
                "seed": int(seed),
                "fold_id": int(fold.fold_id),
                "y_true": int(y_test[position]),
                "y_pred": int(y_pred[position]),
                "y_score": float(y_score[position]),
                "label_true": label_true[position],
                "label_pred": label_pred[position],
            }
        )

    return rows, prediction_rows


def run_loso(
    dataset,
    arm: str,
    seed: int,
    config: dict,
    subject: str | int | None = None,
    max_folds: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run one arm/seed over LOSO folds (or one debug subject)."""
    folds = make_loso_splits(dataset.metadata)
    if subject is not None:
        folds = [fold for fold in folds if str(fold.test_subjects[0]) == str(subject)]
        if not folds:
            available = sorted(
                {str(fold.test_subjects[0]) for fold in make_loso_splits(dataset.metadata)}
            )
            raise ValueError(f"unknown subject {subject!r}; available: {available}")
    if max_folds is not None:
        folds = folds[: int(max_folds)]
    if not folds:
        raise ValueError("no folds selected")

    result_rows: list[dict] = []
    prediction_rows: list[dict] = []
    for fold in folds:
        fold_results, fold_predictions = evaluate_fold(
            dataset=dataset,
            fold=fold,
            arm=arm,
            seed=seed,
            config=config,
        )
        result_rows.extend(fold_results)
        prediction_rows.extend(fold_predictions)

    results = ensure_result_columns(pd.DataFrame(result_rows))
    predictions = ensure_prediction_columns(pd.DataFrame(prediction_rows))
    return results, predictions


def run_all(
    dataset,
    config: dict,
    arms: Iterable[str],
    seeds: Iterable[int],
    subject: str | int | None = None,
    max_folds: int | None = None,
    logger=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every (arm, seed) combination and concatenate the outputs."""
    arm_list = list(arms)
    seed_list = [int(seed) for seed in seeds]
    results_parts: list[pd.DataFrame] = []
    prediction_parts: list[pd.DataFrame] = []

    for arm in arm_list:
        for seed in seed_list:
            if logger is not None:
                logger.log(f"running arm={arm} seed={seed} ...")
            results, predictions = run_loso(
                dataset=dataset,
                arm=arm,
                seed=seed,
                config=config,
                subject=subject,
                max_folds=max_folds,
            )
            results_parts.append(results)
            prediction_parts.append(predictions)
            if logger is not None:
                overall = results[results["session"].astype(str) == "all"]
                logger.log(
                    f"  arm={arm} seed={seed}: "
                    f"mean balanced_acc={overall['balanced_acc'].mean():.4f} "
                    f"({len(overall)} folds)"
                )

    results = pd.concat(results_parts, ignore_index=True) if results_parts else pd.DataFrame()
    predictions = (
        pd.concat(prediction_parts, ignore_index=True)
        if prediction_parts
        else pd.DataFrame()
    )
    return ensure_result_columns(results), ensure_prediction_columns(predictions)


def run_canary(
    dataset,
    arm: str,
    seed: int,
    config: dict,
    subject: str | int | None = None,
    max_folds: int | None = None,
    tolerance: float | None = None,
) -> CanaryResult:
    """Run the label-permutation canary on the M1 pipeline."""

    def run_fn(permuted_dataset) -> float:
        results, _ = run_loso(
            dataset=permuted_dataset,
            arm=arm,
            seed=seed,
            config=config,
            subject=subject,
            max_folds=max_folds,
        )
        overall = results[results["session"].astype(str) == "all"]
        return float(overall["balanced_acc"].mean())

    chance_level = 1.0 / len(dataset.class_names or [])
    if tolerance is None:
        tolerance = float(config.get("evaluation", {}).get("canary_tolerance", 0.10))
    return label_permutation_canary(
        run_fn=run_fn,
        dataset=dataset,
        chance_level=chance_level,
        seed=derive_seed(seed, "canary"),
        tolerance=tolerance,
    )


def dry_run_report(dataset, max_folds: int | None = None) -> pd.DataFrame:
    """Return a fold summary without fitting anything."""
    folds = make_loso_splits(dataset.metadata)
    if max_folds is not None:
        folds = folds[: int(max_folds)]
    rows = []
    for fold in folds:
        test_meta = dataset.metadata.iloc[fold.test_idx]
        rows.append(
            {
                "fold_id": fold.fold_id,
                "test_subject": fold.test_subjects[0],
                "train_subjects": ",".join(fold.train_subjects),
                "n_train": len(fold.train_idx),
                "n_test": len(fold.test_idx),
                "test_sessions": ",".join(
                    sorted(test_meta["session_id"].astype(str).unique().tolist())
                ),
            }
        )
    return pd.DataFrame(rows)


def summary_table(results: pd.DataFrame) -> pd.DataFrame:
    """Compact per-arm/seed aggregate for the console."""
    return aggregate_results(results)
