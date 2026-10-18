"""Subject-grouped LOSO split construction.

The split unit is the subject, never the trial or the window. Every fold is
checked by :func:`common_utils.leakage_checks.run_fold_checks` before it is
returned.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .leakage_checks import run_fold_checks


@dataclass(frozen=True)
class Fold:
    """One LOSO fold.

    ``train_idx`` and ``test_idx`` are positional indices into the metadata /
    feature arrays, not pandas index labels.
    """

    fold_id: int
    train_subjects: tuple[str, ...]
    test_subjects: tuple[str, ...]
    train_idx: np.ndarray
    test_idx: np.ndarray


def make_loso_splits(
    metadata: pd.DataFrame,
    subject_col: str = "subject_id",
) -> list[Fold]:
    """Build one leave-one-subject-out fold per subject and check each fold."""
    subject_values = metadata[subject_col].astype(str).values
    subjects = sorted(set(subject_values.tolist()))
    if len(subjects) < 2:
        raise ValueError(f"LOSO requires at least 2 subjects, got {len(subjects)}")

    folds: list[Fold] = []
    for fold_id, test_subject in enumerate(subjects):
        test_mask = subject_values == test_subject
        train_mask = ~test_mask
        fold = Fold(
            fold_id=fold_id,
            train_subjects=tuple(
                sorted(set(metadata.loc[train_mask, subject_col].astype(str).tolist()))
            ),
            test_subjects=(test_subject,),
            train_idx=np.flatnonzero(train_mask),
            test_idx=np.flatnonzero(test_mask),
        )
        run_fold_checks(metadata, fold)
        folds.append(fold)
    return folds


def make_nested_loso_splits(
    metadata: pd.DataFrame,
    outer_test_subject: str | int,
    subject_col: str = "subject_id",
) -> tuple[np.ndarray, np.ndarray, list[Fold]]:
    """Build outer train/test indices and inner LOSO folds for one outer fold.

    The inner folds are for hyperparameter selection / early stopping only.
    They never include the outer test subject. Returned inner indices are
    positional indices into the *full* metadata.
    """
    outer_test_subject = str(outer_test_subject)
    subject_values = metadata[subject_col].astype(str).values
    outer_test_idx = np.flatnonzero(subject_values == outer_test_subject)
    outer_train_idx = np.flatnonzero(subject_values != outer_test_subject)
    if outer_test_idx.size == 0:
        raise ValueError(f"unknown outer test subject: {outer_test_subject!r}")
    if outer_train_idx.size == 0:
        raise ValueError("no training subjects remain for the outer fold")

    inner_meta = metadata.iloc[outer_train_idx].reset_index(drop=True)
    inner_folds = make_loso_splits(inner_meta, subject_col=subject_col)

    mapped: list[Fold] = []
    for fold in inner_folds:
        mapped.append(
            Fold(
                fold_id=fold.fold_id,
                train_subjects=fold.train_subjects,
                test_subjects=fold.test_subjects,
                train_idx=outer_train_idx[fold.train_idx],
                test_idx=outer_train_idx[fold.test_idx],
            )
        )
    return outer_train_idx, outer_test_idx, mapped


def folds_to_records(folds: list[Fold]) -> list[dict]:
    """Convert folds to a JSON-serializable list."""
    return [
        {
            "fold_id": fold.fold_id,
            "train_subjects": list(fold.train_subjects),
            "test_subjects": list(fold.test_subjects),
            "train_idx": fold.train_idx.tolist(),
            "test_idx": fold.test_idx.tolist(),
        }
        for fold in folds
    ]


def save_folds(folds: list[Fold], path: str | Path) -> Path:
    """Save split indices so any result row can be traced back to its trials."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(folds_to_records(folds), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def load_folds(path: str | Path) -> list[Fold]:
    """Load folds saved by :func:`save_folds`."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        Fold(
            fold_id=item["fold_id"],
            train_subjects=tuple(item["train_subjects"]),
            test_subjects=tuple(item["test_subjects"]),
            train_idx=np.asarray(item["train_idx"], dtype=int),
            test_idx=np.asarray(item["test_idx"], dtype=int),
        )
        for item in payload
    ]
