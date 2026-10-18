"""Dataset loading for M1.

Two sources are supported:

* ``synthetic``: a small, deterministic dataset used for smoke tests and for
  developing the pipeline without downloading MOABB data.
* ``bnci2014_001``: the public motor-imagery dataset via MOABB. MOABB is an
  optional dependency; if it is missing, the CLI prints an actionable error.

Both paths return an :class:`EpochDataset` with the canonical metadata schema
from ``common_utils.metadata``.
"""

from __future__ import annotations

import hashlib
import os
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from common_utils.metadata import (
    metadata_summary,
    normalize_metadata,
    validate_metadata,
)
from common_utils.paths import BNCI_RAW_DIR, ensure_dir, resolve_project_path

BNCI_CLASS_NAMES = ["left_hand", "right_hand", "feet", "tongue"]


@dataclass
class EpochDataset:
    """In-memory epochs plus trial metadata.

    ``X`` has shape ``(n_epochs, n_channels, n_times)``; ``y`` is an integer
    label vector aligned with ``X``; ``metadata`` has one row per epoch.
    """

    X: np.ndarray
    y: np.ndarray
    metadata: pd.DataFrame
    ch_names: list[str]
    sfreq: float
    class_names: list[str] | None = field(default=None)

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=np.float32)
        self.y = np.asarray(self.y)
        self.ch_names = list(self.ch_names)
        validate_metadata(self.metadata)

        if self.X.ndim != 3:
            raise ValueError(f"X must be 3-D (epochs, channels, times), got {self.X.shape}")
        if len(self.X) != len(self.y) or len(self.X) != len(self.metadata):
            raise ValueError(
                "X, y, and metadata must have the same number of epochs: "
                f"{len(self.X)}, {len(self.y)}, {len(self.metadata)}"
            )
        if self.X.shape[1] != len(self.ch_names):
            raise ValueError(
                f"X has {self.X.shape[1]} channels but ch_names has {len(self.ch_names)}"
            )
        if self.class_names is None:
            self.class_names = [str(value) for value in sorted(pd.unique(self.y))]
        else:
            self.class_names = [str(value) for value in self.class_names]

    @property
    def n_epochs(self) -> int:
        return len(self.X)

    @property
    def n_channels(self) -> int:
        return self.X.shape[1]

    @property
    def n_times(self) -> int:
        return self.X.shape[2]

    def subset(self, indices: Any) -> "EpochDataset":
        """Return a positional subset with reset metadata index."""
        indices = np.asarray(indices, dtype=int)
        return EpochDataset(
            X=self.X[indices],
            y=self.y[indices],
            metadata=self.metadata.iloc[indices].reset_index(drop=True),
            ch_names=self.ch_names,
            sfreq=self.sfreq,
            class_names=self.class_names,
        )

    def with_labels(self, labels: Any) -> "EpochDataset":
        """Return a copy with replaced labels (used by the permutation canary)."""
        return EpochDataset(
            X=self.X,
            y=np.asarray(labels),
            metadata=self.metadata.copy(),
            ch_names=self.ch_names,
            sfreq=self.sfreq,
            class_names=self.class_names,
        )

    def to_mne(self):
        """Build an ``mne.EpochsArray`` for IF-1-style exchange."""
        import mne

        info = mne.create_info(self.ch_names, float(self.sfreq), ch_types="eeg")
        y_int = np.asarray(self.y, dtype=int)
        events = np.column_stack(
            [
                np.arange(len(y_int)),
                np.zeros(len(y_int), dtype=int),
                y_int + 1,
            ]
        )
        event_id = {name: index + 1 for index, name in enumerate(self.class_names)}
        epochs = mne.EpochsArray(
            self.X,
            info,
            events=events,
            event_id=event_id,
            tmin=0.0,
            verbose=False,
        )
        epochs.metadata = self.metadata.copy()
        return epochs


def _default_class_names(n_classes: int) -> list[str]:
    if n_classes == len(BNCI_CLASS_NAMES):
        return BNCI_CLASS_NAMES.copy()
    return [f"class_{index}" for index in range(n_classes)]


def load_synthetic_dataset(config: dict) -> EpochDataset:
    """Create a deterministic synthetic dataset with subject-specific offsets.

    The class template gives the classifier something to learn; the subject
    offset simulates identity information. It is a pipeline smoke test, not a
    scientific benchmark.
    """
    cfg = config["dataset"]
    n_subjects = int(cfg.get("n_subjects", 9))
    n_sessions = int(cfg.get("n_sessions", 2))
    n_runs = int(cfg.get("n_runs", 6))
    n_classes = int(cfg.get("n_classes", 4))
    n_trials_per_class = int(cfg.get("n_trials_per_class", 6))
    n_channels = int(cfg.get("n_channels", 22))
    sfreq = float(cfg.get("sfreq", 250.0))
    duration_s = float(cfg.get("duration_s", 1.0))
    seed = int(cfg.get("seed", 0))

    n_times = max(8, int(round(sfreq * duration_s)))
    n_trials_per_run = n_classes * n_trials_per_class
    n_total = n_subjects * n_sessions * n_runs * n_trials_per_run

    rng = np.random.default_rng(seed)
    class_templates = rng.normal(0.0, 0.8, size=(n_classes, n_channels, 1))
    subject_offsets = rng.normal(0.0, 0.5, size=(n_subjects, n_channels, 1))

    X = np.empty((n_total, n_channels, n_times), dtype=np.float32)
    labels = np.empty(n_total, dtype=int)
    rows: list[dict[str, Any]] = []

    cursor = 0
    for subject in range(n_subjects):
        for session in range(n_sessions):
            for run in range(n_runs):
                trial_order = rng.permutation(
                    np.repeat(np.arange(n_classes), n_trials_per_class)
                )
                for trial_index, label in enumerate(trial_order):
                    noise = rng.normal(0.0, 1.0, size=(n_channels, n_times))
                    X[cursor] = class_templates[label] + subject_offsets[subject] + noise
                    labels[cursor] = int(label)
                    rows.append(
                        {
                            "subject_id": f"S{subject + 1:02d}",
                            "session_id": f"ses-{session + 1}",
                            "run_id": f"run-{run + 1}",
                            "trial_id": f"trial-{trial_index + 1:03d}",
                            "label": int(label),
                        }
                    )
                    cursor += 1

    metadata = normalize_metadata(pd.DataFrame(rows), labels=labels)
    ch_names = [f"EEG{index + 1:03d}" for index in range(n_channels)]
    return EpochDataset(
        X=X,
        y=labels,
        metadata=metadata,
        ch_names=ch_names,
        sfreq=sfreq,
        class_names=_default_class_names(n_classes),
    )


def load_bnci2014_001(config: dict) -> EpochDataset:
    """Load BNCI2014-001 through MOABB.

    MOABB handles the fixed band-pass/resampling of the paradigm; the M1
    pipeline still fits all trainable transforms inside each LOSO fold.
    """
    try:
        from moabb.datasets import BNCI2014_001
        from moabb.paradigms import MotorImagery
    except ImportError as exc:
        raise RuntimeError(
            "MOABB is not installed. Install it with `pip install moabb` "
            "or run the smoke test with `--dataset synthetic`."
        ) from exc

    cfg = config["dataset"]

    # Keep the raw BNCI files on the project data disk, not C:. A YAML
    # override is supported via dataset.data_path.
    import mne

    data_path = resolve_project_path(cfg.get("data_path") or BNCI_RAW_DIR)
    ensure_dir(data_path)

    subjects = [int(subject) for subject in cfg.get("subjects", list(range(1, 10)))]
    paradigm = MotorImagery(
        n_classes=int(cfg.get("n_classes", 4)),
        tmin=float(cfg.get("tmin", 0.0)),
        tmax=float(cfg.get("tmax", 4.0)),
        resample=cfg.get("resample", 250),
        fmin=float(cfg.get("fmin", 8.0)),
        fmax=float(cfg.get("fmax", 32.0)),
    )
    dataset = BNCI2014_001()

    # MOABB 1.5's _sanitize_path replaces ':' in absolute Windows paths,
    # which creates a bogus "D:\\..." tree. Use "." as the BNCI root while
    # running from the data directory, then restore cwd and MNE config.
    previous_cwd = os.getcwd()
    previous_bnci_path = mne.get_config("MNE_DATASETS_BNCI_PATH")
    try:
        os.chdir(data_path)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Setting non-standard config type")
            mne.set_config("MNE_DATASETS_BNCI_PATH", ".")
        X, labels, metadata = paradigm.get_data(dataset=dataset, subjects=subjects)
    finally:
        os.chdir(previous_cwd)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Setting non-standard config type")
            mne.set_config("MNE_DATASETS_BNCI_PATH", previous_bnci_path)

    if hasattr(X, "get_data"):
        X = X.get_data(copy=False)
    X = np.asarray(X, dtype=np.float32)

    metadata = normalize_metadata(metadata, labels=labels)
    encoder = LabelEncoder()
    y = encoder.fit_transform(metadata["label"].astype(str))

    ch_names = list(getattr(paradigm, "channels", []) or [])
    if len(ch_names) != X.shape[1]:
        ch_names = [f"EEG{index + 1:03d}" for index in range(X.shape[1])]

    sfreq = float(cfg.get("resample") or 250.0)
    return EpochDataset(
        X=X,
        y=y,
        metadata=metadata,
        ch_names=ch_names,
        sfreq=sfreq,
        class_names=[str(name) for name in encoder.classes_],
    )


def load_epochs_fif(path: str, config: dict | None = None) -> EpochDataset:
    """Load an IF-1 ``mne.Epochs`` file with metadata.

    The ``.fif`` file must carry ``epochs.metadata`` with (at least) a label
    column; subject/session/run/trial columns are normalized if present.
    """
    import mne

    epochs = mne.read_epochs(path, preload=True, verbose=False)
    if epochs.metadata is None:
        raise ValueError(
            f"{path} has no epochs.metadata; IF-1 requires a label column and "
            "subject/session/run/trial identifiers"
        )
    metadata = normalize_metadata(epochs.metadata)
    encoder = LabelEncoder()
    y = encoder.fit_transform(metadata["label"].astype(str))
    X = epochs.get_data(copy=True)
    return EpochDataset(
        X=X,
        y=y,
        metadata=metadata,
        ch_names=list(epochs.ch_names),
        sfreq=float(epochs.info["sfreq"]),
        class_names=[str(name) for name in encoder.classes_],
    )


def load_dataset(name: str, config: dict) -> EpochDataset:
    """Dispatch to the requested dataset loader."""
    if name == "synthetic":
        return load_synthetic_dataset(config)
    if name == "bnci2014_001":
        return load_bnci2014_001(config)
    raise ValueError(f"unknown dataset: {name!r}")


def dataset_summary(dataset: EpochDataset) -> dict:
    """JSON-serializable dataset summary for run logs."""
    summary = metadata_summary(dataset.metadata)
    summary.update(
        {
            "shape": list(dataset.X.shape),
            "ch_names": dataset.ch_names,
            "sfreq": dataset.sfreq,
            "class_names": dataset.class_names,
        }
    )
    return summary


def dataset_fingerprint(dataset: EpochDataset, max_epochs: int = 128) -> str:
    """Cheap, stable fingerprint for the run log.

    Hashing every epoch would be slow for large datasets; the first
    ``max_epochs`` plus labels and metadata are enough to detect accidental
    data swaps in day-to-day runs.
    """
    digest = hashlib.sha256()
    sample = dataset.X[: min(max_epochs, len(dataset.X))]
    digest.update(np.asarray(sample, dtype=np.float32).tobytes())
    digest.update(np.asarray(dataset.y, dtype=np.int64).tobytes())
    digest.update(
        dataset.metadata.to_csv(index=False).encode("utf-8")
    )
    return digest.hexdigest()[:16]


def save_epochs_fif(dataset: EpochDataset, path: str) -> None:
    """Save an IF-1-compatible ``mne.Epochs`` file."""
    dataset.to_mne().save(path, overwrite=True, verbose=False)
