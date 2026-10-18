"""Trial-level metadata used to enforce subject/trial grouping.

The canonical schema is::

    epoch_id, subject_id, session_id, run_id, trial_id, label

``epoch_id`` is unique per model input. ``trial_id`` is the original trial and
may repeat when one trial is cut into overlapping windows; all rows sharing a
``trial_id`` must stay in the same split.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

REQUIRED_METADATA_COLUMNS = [
    "epoch_id",
    "subject_id",
    "session_id",
    "run_id",
    "trial_id",
    "label",
]

TRIAL_KEY = ["subject_id", "session_id", "run_id", "trial_id"]

_RENAME = {
    "subject": "subject_id",
    "subject_id": "subject_id",
    "session": "session_id",
    "session_id": "session_id",
    "run": "run_id",
    "run_id": "run_id",
    "trial": "trial_id",
    "trial_id": "trial_id",
    "class": "label",
    "label": "label",
    "y": "label",
}


def normalize_metadata(metadata: pd.DataFrame, labels: Any | None = None) -> pd.DataFrame:
    """Normalize common MOABB/MNE column names and validate the schema.

    Parameters
    ----------
    metadata:
        DataFrame returned by MOABB or read from ``mne.Epochs.metadata``.
    labels:
        Optional label vector overriding ``metadata['label']``.
    """
    if not isinstance(metadata, pd.DataFrame):
        raise TypeError(f"metadata must be a pandas DataFrame, got {type(metadata)!r}")

    df = metadata.copy()
    df = df.rename(columns={key: value for key, value in _RENAME.items() if key in df.columns})

    for column in ("subject_id", "session_id", "run_id"):
        if column not in df.columns:
            if column == "run_id":
                df[column] = 0
            elif column == "session_id":
                df[column] = 0
            else:
                raise ValueError(f"metadata is missing required column {column!r}")

    if "trial_id" not in df.columns:
        # No trial identifier was provided: use a per-run sequence. This keeps
        # each epoch distinct and makes the trial-grouping check meaningful.
        df["trial_id"] = (
            df.groupby(["subject_id", "session_id", "run_id"], sort=False)
            .cumcount()
            .astype(str)
        )

    if labels is not None:
        if len(labels) != len(df):
            raise ValueError(f"labels length {len(labels)} != metadata length {len(df)}")
        df["label"] = labels

    if "label" not in df.columns:
        raise ValueError("metadata is missing a label column (expected 'label', 'class', or 'y')")

    for column in ("subject_id", "session_id", "run_id", "trial_id"):
        df[column] = df[column].astype(str)

    if "epoch_id" not in df.columns:
        base = (
            df["subject_id"]
            + "|"
            + df["session_id"]
            + "|"
            + df["run_id"]
            + "|"
            + df["trial_id"]
        )
        duplicated = base.duplicated(keep=False)
        if duplicated.any():
            counter = base.groupby(base).cumcount().astype(str)
            base = base.where(~duplicated, base + "#" + counter)
        df["epoch_id"] = base
    else:
        df["epoch_id"] = df["epoch_id"].astype(str)

    df = df.reset_index(drop=True)
    validate_metadata(df)
    return df


def validate_metadata(df: pd.DataFrame, require_unique_epochs: bool = True) -> pd.DataFrame:
    """Raise ``ValueError`` if the canonical metadata schema is violated."""
    missing = [column for column in REQUIRED_METADATA_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"metadata is missing required columns: {missing}")

    if df[REQUIRED_METADATA_COLUMNS].isnull().any().any():
        bad = df[REQUIRED_METADATA_COLUMNS].isnull().sum()
        raise ValueError(f"metadata contains null values:\n{bad[bad > 0]}")

    if require_unique_epochs and df["epoch_id"].duplicated().any():
        duplicates = df.loc[df["epoch_id"].duplicated(), "epoch_id"].tolist()[:5]
        raise ValueError(f"epoch_id must be unique; examples: {duplicates}")

    return df


def metadata_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Small JSON-serializable summary for run logs."""
    subjects = sorted(df["subject_id"].unique().tolist())
    return {
        "n_epochs": int(len(df)),
        "n_subjects": int(len(subjects)),
        "subjects": subjects,
        "sessions_per_subject": {
            subject: sorted(df.loc[df["subject_id"] == subject, "session_id"].unique().tolist())
            for subject in subjects
        },
        "n_sessions": int(df["session_id"].nunique()),
        "n_runs": int(df["run_id"].nunique()),
        "n_labels": int(df["label"].nunique()),
        "label_counts": {
            str(label): int(count)
            for label, count in df["label"].value_counts().sort_index().items()
        },
    }
