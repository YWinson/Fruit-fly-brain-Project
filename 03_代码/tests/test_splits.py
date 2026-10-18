"""Split and hard-assertion tests."""

from __future__ import annotations

import pandas as pd
import pytest

from common_utils.leakage_checks import (
    LeakageError,
    assert_all_subject_sessions_in_test,
    assert_subject_disjoint,
    assert_trial_grouped,
)
from common_utils.metadata import normalize_metadata
from common_utils.splits import load_folds, make_loso_splits, save_folds


def make_metadata(
    n_subjects: int = 3,
    n_sessions: int = 2,
    n_runs: int = 2,
    n_trials: int = 2,
) -> pd.DataFrame:
    rows = []
    for subject in range(n_subjects):
        for session in range(n_sessions):
            for run in range(n_runs):
                for trial in range(n_trials):
                    rows.append(
                        {
                            "subject_id": f"S{subject + 1:02d}",
                            "session_id": f"ses-{session + 1}",
                            "run_id": f"run-{run + 1}",
                            "trial_id": f"trial-{trial + 1}",
                            "label": (subject + trial) % 2,
                        }
                    )
    return normalize_metadata(pd.DataFrame(rows))


def test_loso_makes_one_fold_per_subject_and_holds_out_all_sessions():
    metadata = make_metadata()
    folds = make_loso_splits(metadata)

    assert len(folds) == 3
    assert [fold.test_subjects[0] for fold in folds] == ["S01", "S02", "S03"]

    for fold in folds:
        train = metadata.iloc[fold.train_idx]
        test = metadata.iloc[fold.test_idx]
        assert set(train["subject_id"]).isdisjoint(set(test["subject_id"]))
        assert set(test["session_id"]) == {"ses-1", "ses-2"}
        assert test["subject_id"].nunique() == 1


def test_subject_disjoint_assertion_detects_overlap():
    metadata = make_metadata()
    with pytest.raises(LeakageError):
        assert_subject_disjoint(metadata, metadata)


def test_session_completeness_detects_missing_session():
    metadata = make_metadata()
    test_meta = metadata[
        (metadata["subject_id"] == "S01") & (metadata["session_id"] == "ses-1")
    ]
    with pytest.raises(LeakageError):
        assert_all_subject_sessions_in_test(metadata, test_meta)


def test_trial_grouped_detects_a_trial_spanning_splits():
    metadata = make_metadata()
    fold_meta = metadata.copy()
    fold_meta["split"] = "train"
    duplicated_trial = fold_meta.iloc[[0]].copy()
    duplicated_trial["split"] = "test"
    fold_meta = pd.concat([fold_meta, duplicated_trial], ignore_index=True)

    with pytest.raises(LeakageError):
        assert_trial_grouped(fold_meta)


def test_save_and_load_folds_roundtrip(tmp_path):
    metadata = make_metadata()
    folds = make_loso_splits(metadata)
    path = tmp_path / "splits.json"
    save_folds(folds, path)
    loaded = load_folds(path)

    assert len(loaded) == len(folds)
    for original, restored in zip(folds, loaded):
        assert original.train_subjects == restored.train_subjects
        assert original.test_subjects == restored.test_subjects
        assert original.train_idx.tolist() == restored.train_idx.tolist()
        assert original.test_idx.tolist() == restored.test_idx.tolist()


def test_missing_trial_id_is_generated_per_run():
    metadata = normalize_metadata(
        pd.DataFrame(
            {
                "subject": ["S1"] * 4,
                "session": ["ses-1"] * 4,
                "run": ["run-1"] * 4,
                "label": [0, 1, 0, 1],
            }
        )
    )
    assert metadata["trial_id"].nunique() == 4
    assert metadata["epoch_id"].is_unique
