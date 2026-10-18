"""Smoke tests for the M1 pipeline (no MOABB required)."""

from __future__ import annotations

import copy

import yaml

from common_utils.config import DEFAULT_CONFIG
from common_utils.leakage_checks import CanaryResult
from common_utils.results import RESULT_COLUMNS
from M1_评测管线.data import load_synthetic_dataset
from M1_评测管线.evaluate import dry_run_report, run_canary, run_loso
from M1_评测管线.run_m1 import main as cli_main


def make_config() -> dict:
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["dataset"].update(
        {
            "name": "synthetic",
            "n_subjects": 3,
            "n_sessions": 2,
            "n_runs": 2,
            "n_classes": 2,
            "n_trials_per_class": 4,
            "n_channels": 8,
            "sfreq": 100.0,
            "duration_s": 1.0,
            "seed": 0,
        }
    )
    config["preprocess"]["csp_components"] = 4
    config["evaluation"]["seeds"] = [0]
    config["evaluation"]["canary_tolerance"] = 0.25
    config["arms"] = ["baseline", "equiv_dropout"]
    return config


def test_loso_smoke_baseline_and_equiv_dropout():
    config = make_config()
    dataset = load_synthetic_dataset(config)

    results, predictions = run_loso(
        dataset=dataset,
        arm="baseline",
        seed=0,
        config=config,
        max_folds=1,
    )
    assert set(RESULT_COLUMNS).issubset(results.columns)
    assert len(predictions) == 32  # 2 sessions x 2 runs x 2 classes x 4 trials
    assert results["balanced_acc"].between(0.0, 1.0).all()
    assert (results["session"].astype(str) == "all").sum() == 1

    dropout_results, _ = run_loso(
        dataset=dataset,
        arm="equiv_dropout",
        seed=0,
        config=config,
        max_folds=1,
    )
    assert (dropout_results["mask_id"].astype(str).str.len() > 0).all()


def test_dry_run_report_and_canary_type():
    config = make_config()
    dataset = load_synthetic_dataset(config)

    report = dry_run_report(dataset, max_folds=1)
    assert len(report) == 1
    assert report.loc[0, "n_test"] == 32

    canary = run_canary(
        dataset=dataset,
        arm="baseline",
        seed=0,
        config=config,
        max_folds=1,
    )
    assert isinstance(canary, CanaryResult)


def test_cli_dry_run(tmp_path, capsys):
    config = make_config()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    exit_code = cli_main(
        [
            "--config",
            str(config_path),
            "--dataset",
            "synthetic",
            "--arm",
            "baseline",
            "--seed",
            "0",
            "--dry-run",
            "--max-folds",
            "1",
            "--log-dir",
            str(tmp_path / "logs"),
            "--results-dir",
            str(tmp_path / "results"),
        ]
    )
    captured = capsys.readouterr()
    assert exit_code == 0, captured.out + captured.err
    assert "fold_id" in captured.out
