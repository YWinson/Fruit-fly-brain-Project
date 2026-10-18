"""Hard leakage checks.

These are deliberately exceptions, not warnings: a failed check must stop the
run. The M1 checklist (``06_会议与笔记/M1_LOSO_泄露自查清单.md``) refers to
these functions as the code-level enforcement of items A2/A4/A5/A10-E1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from .metadata import TRIAL_KEY


class LeakageError(AssertionError):
    """Raised when a split or pipeline violates a leakage invariant."""


def _string_set(values: Sequence[Any]) -> set[str]:
    return {str(value) for value in values}


def assert_subject_disjoint(
    train_meta: pd.DataFrame,
    test_meta: pd.DataFrame,
    subject_col: str = "subject_id",
) -> None:
    """Assert that no subject appears in both train and test metadata."""
    train_subjects = _string_set(train_meta[subject_col].tolist())
    test_subjects = _string_set(test_meta[subject_col].tolist())
    overlap = train_subjects & test_subjects
    if overlap:
        raise LeakageError(
            "subject leakage: subjects present in both train and test: "
            f"{sorted(overlap)}"
        )


def assert_all_subject_sessions_in_test(
    full_meta: pd.DataFrame,
    test_meta: pd.DataFrame,
    subject_col: str = "subject_id",
    session_col: str = "session_id",
) -> None:
    """Assert that every session of each test subject is held out.

    For BNCI2014-001 this means both sessions of the held-out subject must be
    in the test partition; a train/test split by session is *not* LOSO.
    """
    for subject in _string_set(test_meta[subject_col].tolist()):
        expected = _string_set(
            full_meta.loc[
                full_meta[subject_col].astype(str) == subject, session_col
            ].tolist()
        )
        actual = _string_set(
            test_meta.loc[
                test_meta[subject_col].astype(str) == subject, session_col
            ].tolist()
        )
        if actual != expected:
            missing = expected - actual
            raise LeakageError(
                f"session leakage: subject {subject!r} has sessions {sorted(expected)} "
                f"but test contains {sorted(actual)}; missing {sorted(missing)}"
            )


def assert_trial_grouped(
    fold_meta: pd.DataFrame,
    trial_key: Sequence[str] = TRIAL_KEY,
    split_col: str = "split",
) -> None:
    """Assert that every trial belongs to exactly one split.

    This catches overlapping-window leakage: if windows from one trial are
    randomly split, the same trial appears in both ``train`` and ``test``.
    """
    if split_col not in fold_meta.columns:
        raise ValueError(f"fold_meta must contain a {split_col!r} column")
    grouped = fold_meta.groupby(list(trial_key), dropna=False)[split_col].nunique()
    bad = grouped[grouped > 1]
    if len(bad):
        examples = bad.index.tolist()[:5]
        raise LeakageError(
            f"trial-level leakage: {len(bad)} trial(s) span multiple splits; "
            f"examples: {examples}"
        )


def assert_no_duplicate_epochs(
    meta: pd.DataFrame,
    epoch_col: str = "epoch_id",
) -> None:
    """Assert that ``epoch_id`` is unique."""
    duplicated = meta[epoch_col].duplicated()
    if duplicated.any():
        examples = meta.loc[duplicated, epoch_col].tolist()[:5]
        raise LeakageError(f"duplicate {epoch_col} values; examples: {examples}")


def assert_train_test_indices_disjoint(
    train_idx: Sequence[int],
    test_idx: Sequence[int],
) -> None:
    """Assert that positional train/test index arrays do not overlap."""
    overlap = np.intersect1d(np.asarray(train_idx), np.asarray(test_idx))
    if overlap.size:
        raise LeakageError(f"train/test index overlap: {overlap[:10].tolist()}")


def assert_no_test_subject_in_fit(
    fit_subjects: Sequence[Any],
    test_subjects: Sequence[Any],
) -> None:
    """Assert that no test subject is used to fit any transform or model."""
    overlap = _string_set(fit_subjects) & _string_set(test_subjects)
    if overlap:
        raise LeakageError(f"test subject used during fit: {sorted(overlap)}")


def run_fold_checks(metadata: pd.DataFrame, fold: Any) -> None:
    """Run every split-level check for one fold.

    ``fold`` is expected to expose ``train_idx``, ``test_idx``,
    ``train_subjects``, and ``test_subjects`` (the ``Fold`` dataclass in
    ``common_utils.splits``).
    """
    train_idx = np.asarray(fold.train_idx, dtype=int)
    test_idx = np.asarray(fold.test_idx, dtype=int)

    assert_train_test_indices_disjoint(train_idx, test_idx)
    assert_no_duplicate_epochs(metadata)

    train_meta = metadata.iloc[train_idx]
    test_meta = metadata.iloc[test_idx]

    assert_subject_disjoint(train_meta, test_meta)
    assert_all_subject_sessions_in_test(metadata, test_meta)
    assert_no_test_subject_in_fit(
        train_meta["subject_id"].tolist(),
        test_meta["subject_id"].tolist(),
    )

    split = np.array(["train"] * len(metadata), dtype=object)
    split[test_idx] = "test"
    fold_meta = metadata.copy()
    fold_meta["split"] = split
    assert_trial_grouped(fold_meta)


@dataclass(frozen=True)
class CanaryResult:
    """Result of a label-permutation leakage canary."""

    accuracy: float
    chance_level: float
    tolerance: float
    passed: bool
    message: str


def label_permutation_canary(
    run_fn: Callable[[Any], float],
    dataset: Any,
    chance_level: float,
    seed: int = 0,
    tolerance: float = 0.10,
) -> CanaryResult:
    """Permute labels globally and check that performance falls to chance.

    ``run_fn`` receives a dataset-like object with ``with_labels(y)`` and must
    return a single balanced-accuracy value. The canary is a diagnostic, not a
    proof: with very small samples the estimate is noisy.
    """
    y = np.asarray(dataset.y)
    permuted_labels = np.random.default_rng(int(seed)).permutation(y)
    permuted_dataset = dataset.with_labels(permuted_labels)
    accuracy = float(run_fn(permuted_dataset))

    passed = bool(np.isfinite(accuracy) and accuracy <= chance_level + tolerance)
    if passed:
        message = (
            f"canary passed: permuted-label balanced accuracy {accuracy:.3f} "
            f"<= chance {chance_level:.3f} + tolerance {tolerance:.3f}"
        )
    else:
        message = (
            "canary FAILED: permuted-label balanced accuracy "
            f"{accuracy:.3f} > chance {chance_level:.3f} + tolerance "
            f"{tolerance:.3f}; inspect preprocessing/model-selection leakage"
        )
    return CanaryResult(
        accuracy=accuracy,
        chance_level=float(chance_level),
        tolerance=float(tolerance),
        passed=passed,
        message=message,
    )
