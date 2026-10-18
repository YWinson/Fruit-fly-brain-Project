"""Label-permutation canary tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from common_utils.leakage_checks import label_permutation_canary
from common_utils.metadata import normalize_metadata
from M1_评测管线.data import EpochDataset


def make_dataset(n_per_class: int = 20, n_classes: int = 2, seed: int = 0) -> EpochDataset:
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(n_classes), n_per_class)
    n_epochs = len(y)
    X = rng.normal(0.0, 1.0, size=(n_epochs, 4, 8)).astype(np.float32)
    X[y == 0, 0, 0] += 2.0
    if n_classes > 1:
        X[y == 1, 0, 0] -= 2.0

    rows = [
        {
            "subject_id": f"S{index // 10:02d}",
            "session_id": "ses-1",
            "run_id": "run-1",
            "trial_id": f"trial-{index:03d}",
            "label": int(label),
        }
        for index, label in enumerate(y)
    ]
    metadata = normalize_metadata(pd.DataFrame(rows), labels=y)
    return EpochDataset(
        X=X,
        y=y,
        metadata=metadata,
        ch_names=[f"ch{index}" for index in range(X.shape[1])],
        sfreq=100.0,
        class_names=[str(index) for index in range(n_classes)],
    )


def leaky_run_fn(dataset) -> float:
    """A deliberately leaky 'pipeline': fit and predict on the same rows."""
    X = dataset.X.reshape(len(dataset.X), -1)
    model = KNeighborsClassifier(n_neighbors=1).fit(X, dataset.y)
    predictions = model.predict(X)
    return float(balanced_accuracy_score(dataset.y, predictions))


def honest_run_fn(dataset) -> float:
    return 0.5


def test_canary_flags_a_memorising_pipeline():
    dataset = make_dataset()
    result = label_permutation_canary(
        run_fn=leaky_run_fn,
        dataset=dataset,
        chance_level=0.5,
        seed=0,
        tolerance=0.10,
    )
    assert not result.passed
    assert result.accuracy > 0.9


def test_canary_passes_an_at_chance_pipeline():
    dataset = make_dataset()
    result = label_permutation_canary(
        run_fn=honest_run_fn,
        dataset=dataset,
        chance_level=0.5,
        seed=0,
        tolerance=0.10,
    )
    assert result.passed
    assert result.accuracy == 0.5
