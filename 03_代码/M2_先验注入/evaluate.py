"""M2 LOSO evaluation with shared Stage 1 features.

For each fold the multi-band CSP + scaler is fitted once on the training
subjects; all arms then train only their masked MLP on the cached 32-D
features. This is the same statistical procedure as refitting Stage 1 per arm,
but it avoids 5x redundant CSP work.
"""

from __future__ import annotations

import json
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from common_utils.leakage_checks import CanaryResult, label_permutation_canary, run_fold_checks
from common_utils.results import (
    aggregate_results,
    ensure_prediction_columns,
    ensure_result_columns,
)
from common_utils.seeding import derive_seed, set_global_seed
from common_utils.splits import Fold, make_loso_splits

from .arms import MODEL_NAMES, build_classifier, build_stage1
from .clusters import GranularityResult
from .masks import MaskError


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


def _sanitize_mask_info(info: dict) -> dict:
    drop = {
        "out_degree",
        "in_degree",
        "out_degree_sequence",
        "in_degree_sequence",
    }
    return {key: value for key, value in info.items() if key not in drop}


def _result_rows_for_arm(
    dataset,
    test_meta: pd.DataFrame,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    arm: str,
    seed: int,
    fold_id: int,
    mask_info: dict,
    mask_path: str,
    stage1_fit_time_s: float,
    stage1_transform_time_s: float,
    clf_fit_time_s: float,
    latency_ms: float,
    memory_mb: float,
    config: dict,
    n_train: int,
) -> tuple[list[dict], list[dict]]:
    subject = str(test_meta["subject_id"].iloc[0])
    note = json.dumps(
        {
            "dataset": config.get("dataset", {}).get("name"),
            "fold_id": int(fold_id),
            "n_train": int(n_train),
            "n_test": int(len(y_test)),
            "stage1_fit_time_s": round(float(stage1_fit_time_s), 6),
            "stage1_transform_time_s": round(float(stage1_transform_time_s), 6),
            "clf_fit_time_s": round(float(clf_fit_time_s), 6),
            "mask": _sanitize_mask_info(mask_info),
            "mask_path": mask_path,
        },
        ensure_ascii=False,
    )

    rows: list[dict] = []
    session_values = test_meta["session_id"].astype(str).values
    for session in sorted(set(session_values.tolist())):
        mask_session = session_values == session
        rows.append(
            {
                "arm": arm,
                "model": MODEL_NAMES[arm],
                "subject": subject,
                "session": session,
                "seed": int(seed),
                "balanced_acc": _safe_balanced_accuracy(
                    y_test[mask_session], y_pred[mask_session]
                ),
                "latency_ms": latency_ms,
                "memory_MB": memory_mb,
                "note": note,
                "ablation_target": "",
                "mask_id": str(mask_info.get("mask_id", "")),
                "activation_map_path": mask_path,
            }
        )
    rows.append(
        {
            "arm": arm,
            "model": MODEL_NAMES[arm],
            "subject": subject,
            "session": "all",
            "seed": int(seed),
            "balanced_acc": _safe_balanced_accuracy(y_test, y_pred),
            "latency_ms": latency_ms,
            "memory_MB": memory_mb,
            "note": note,
            "ablation_target": "",
            "mask_id": str(mask_info.get("mask_id", "")),
            "activation_map_path": mask_path,
        }
    )

    prediction_rows: list[dict] = []
    label_true = _names_for(y_test, list(dataset.class_names or []))
    label_pred = _names_for(y_pred, list(dataset.class_names or []))
    for position in range(len(y_test)):
        prediction_rows.append(
            {
                "arm": arm,
                "model": MODEL_NAMES[arm],
                "subject": str(test_meta.loc[position, "subject_id"]),
                "session": str(test_meta.loc[position, "session_id"]),
                "run": str(test_meta.loc[position, "run_id"]),
                "trial": str(test_meta.loc[position, "trial_id"]),
                "epoch_id": str(test_meta.loc[position, "epoch_id"]),
                "seed": int(seed),
                "fold_id": int(fold_id),
                "y_true": int(y_test[position]),
                "y_pred": int(y_pred[position]),
                "y_score": float(y_score[position]),
                "label_true": label_true[position],
                "label_pred": label_pred[position],
            }
        )
    return rows, prediction_rows


def evaluate_fold_for_arms(
    dataset,
    fold: Fold,
    arms: list[str],
    seed: int,
    config: dict,
    granularity_result: GranularityResult,
    mask_dir: str | Path | None = None,
    logger=None,
) -> tuple[list[dict], list[dict], list[str]]:
    """Fit Stage 1 once, then every arm's masked MLP, for one fold/seed."""
    run_fold_checks(dataset.metadata, fold)

    train_idx = np.asarray(fold.train_idx, dtype=int)
    test_idx = np.asarray(fold.test_idx, dtype=int)
    X_train, y_train = dataset.X[train_idx], dataset.y[train_idx]
    X_test, y_test = dataset.X[test_idx], dataset.y[test_idx]

    stage1 = build_stage1(config, seed=seed, fold_id=fold.fold_id)
    tracemalloc.start()
    fit_start = time.perf_counter()
    X_train_feat = stage1.fit_transform(X_train, y_train)
    stage1_fit_time_s = time.perf_counter() - fit_start
    transform_start = time.perf_counter()
    X_test_feat = stage1.transform(X_test)
    stage1_transform_time_s = time.perf_counter() - transform_start
    _, stage1_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    test_meta = dataset.metadata.iloc[test_idx].reset_index(drop=True)
    result_rows: list[dict] = []
    prediction_rows: list[dict] = []
    skipped: list[str] = []

    for arm in arms:
        try:
            classifier, mask, sign, mask_info = build_classifier(
                arm,
                config,
                seed=seed,
                fold_id=fold.fold_id,
                granularity_result=granularity_result,
            )
        except MaskError as exc:
            skipped.append(f"{arm}: {exc}")
            if logger is not None:
                logger.log(f"  skipping arm={arm}: {exc}")
            continue

        mask_path = ""
        if mask_dir is not None:
            mask_dir = Path(mask_dir)
            mask_dir.mkdir(parents=True, exist_ok=True)
            path = mask_dir / f"{arm}_fold{fold.fold_id}_seed{seed}.npz"
            np.savez_compressed(path, mask=mask, sign=sign)
            mask_path = str(path)

        set_global_seed(derive_seed(seed, fold.fold_id, arm, "m2_fit"))
        tracemalloc.start()
        clf_start = time.perf_counter()
        classifier.fit(X_train_feat, y_train)
        clf_fit_time_s = time.perf_counter() - clf_start
        predict_start = time.perf_counter()
        y_pred = np.asarray(classifier.predict(X_test_feat))
        predict_time_s = time.perf_counter() - predict_start
        probabilities = np.asarray(classifier.predict_proba(X_test_feat))
        y_score = probabilities.max(axis=1)
        _, clf_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        latency_ms = (
            (stage1_transform_time_s + predict_time_s) / max(1, len(y_test))
        ) * 1000.0
        memory_mb = max(stage1_peak, clf_peak) / (1024.0**2)
        rows, predictions = _result_rows_for_arm(
            dataset=dataset,
            test_meta=test_meta,
            y_test=y_test,
            y_pred=y_pred,
            y_score=y_score,
            arm=arm,
            seed=seed,
            fold_id=fold.fold_id,
            mask_info=mask_info,
            mask_path=mask_path,
            stage1_fit_time_s=stage1_fit_time_s,
            stage1_transform_time_s=stage1_transform_time_s,
            clf_fit_time_s=clf_fit_time_s,
            latency_ms=latency_ms,
            memory_mb=memory_mb,
            config=config,
            n_train=len(train_idx),
        )
        result_rows.extend(rows)
        prediction_rows.extend(predictions)

    return result_rows, prediction_rows, skipped


def run_loso(
    dataset,
    arm: str,
    seed: int,
    config: dict,
    granularity_result: GranularityResult,
    subject: str | int | None = None,
    max_folds: int | None = None,
    mask_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    folds = make_loso_splits(dataset.metadata)
    if subject is not None:
        folds = [fold for fold in folds if str(fold.test_subjects[0]) == str(subject)]
        if not folds:
            raise ValueError(f"unknown subject: {subject!r}")
    if max_folds is not None:
        folds = folds[: int(max_folds)]
    if not folds:
        raise ValueError("no folds selected")

    result_rows: list[dict] = []
    prediction_rows: list[dict] = []
    for fold in folds:
        rows, predictions, _ = evaluate_fold_for_arms(
            dataset=dataset,
            fold=fold,
            arms=[arm],
            seed=seed,
            config=config,
            granularity_result=granularity_result,
            mask_dir=mask_dir,
        )
        result_rows.extend(rows)
        prediction_rows.extend(predictions)
    return (
        ensure_result_columns(pd.DataFrame(result_rows)),
        ensure_prediction_columns(pd.DataFrame(prediction_rows)),
    )


def run_all(
    dataset,
    config: dict,
    arms: list[str],
    seeds: list[int],
    granularity_result: GranularityResult,
    subject: str | int | None = None,
    max_folds: int | None = None,
    mask_dir: str | Path | None = None,
    logger=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    folds = make_loso_splits(dataset.metadata)
    if subject is not None:
        folds = [fold for fold in folds if str(fold.test_subjects[0]) == str(subject)]
        if not folds:
            raise ValueError(f"unknown subject: {subject!r}")
    if max_folds is not None:
        folds = folds[: int(max_folds)]
    if not folds:
        raise ValueError("no folds selected")

    result_rows: list[dict] = []
    prediction_rows: list[dict] = []
    skipped: list[str] = []

    for seed in seeds:
        if logger is not None:
            logger.log(f"running M2 seed={seed} ({len(folds)} folds, {len(arms)} arms) ...")
        for fold in folds:
            rows, predictions, fold_skipped = evaluate_fold_for_arms(
                dataset=dataset,
                fold=fold,
                arms=arms,
                seed=seed,
                config=config,
                granularity_result=granularity_result,
                mask_dir=mask_dir,
                logger=logger,
            )
            result_rows.extend(rows)
            prediction_rows.extend(predictions)
            skipped.extend(fold_skipped)

        partial = ensure_result_columns(pd.DataFrame(result_rows))
        overall = partial[partial["session"].astype(str) == "all"]
        if logger is not None:
            for arm in arms:
                values = overall[overall["arm"] == arm]["balanced_acc"]
                if len(values):
                    logger.log(
                        f"  seed={seed} arm={arm}: mean={values.mean():.4f} "
                        f"({len(values)} folds)"
                    )

    results = (
        ensure_result_columns(pd.DataFrame(result_rows))
        if result_rows
        else ensure_result_columns(pd.DataFrame())
    )
    predictions = (
        ensure_prediction_columns(pd.DataFrame(prediction_rows))
        if prediction_rows
        else ensure_prediction_columns(pd.DataFrame())
    )
    if logger is not None and skipped:
        logger.log(f"skipped {len(set(skipped))} arm/fold combinations")
    return results, predictions


def run_canary(
    dataset,
    arm: str,
    seed: int,
    config: dict,
    granularity_result: GranularityResult,
    subject: str | int | None = None,
    max_folds: int | None = None,
    tolerance: float | None = None,
) -> CanaryResult:
    def run_fn(permuted_dataset) -> float:
        results, _ = run_loso(
            dataset=permuted_dataset,
            arm=arm,
            seed=seed,
            config=config,
            granularity_result=granularity_result,
            subject=subject,
            max_folds=max_folds,
            mask_dir=None,
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
        seed=derive_seed(seed, "m2_canary"),
        tolerance=tolerance,
    )


def fold_report(dataset, max_folds: int | None = None) -> pd.DataFrame:
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
    return aggregate_results(results)
