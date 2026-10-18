"""Configuration loading and validation for the M1 pipeline."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

ALLOWED_DATASETS = {"synthetic", "bnci2014_001"}
ALLOWED_ARMS = {"baseline", "equiv_dropout"}

DEFAULT_CONFIG: dict = {
    "dataset": {
        "name": "synthetic",
        "n_subjects": 9,
        "n_sessions": 2,
        "n_runs": 6,
        "n_classes": 4,
        "n_trials_per_class": 12,
        "n_channels": 22,
        "sfreq": 250.0,
        "duration_s": 2.0,
        "seed": 0,
        # BNCI2014-001-only fields
        "subjects": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "data_path": "02_数据/BNCI2014-001_原始",
        "tmin": 0.0,
        "tmax": 4.0,
        "resample": 250,
        "fmin": 8.0,
        "fmax": 32.0,
    },
    "preprocess": {
        "standardize": True,
        "csp_components": 6,
        "csp_reg": None,
        "csp_log": True,
    },
    "model": {
        "type": "lda",
        "lda_solver": "lsqr",
        "lda_shrinkage": "auto",
    },
    "arms": ["baseline", "equiv_dropout"],
    "equiv_dropout": {
        "density": 0.5,
    },
    "evaluation": {
        "seeds": [0, 1, 2],
        "metrics": ["balanced_accuracy"],
        "save_predictions": True,
        "canary_tolerance": 0.10,
    },
    "output": {
        "results_dir": "04_实验结果/结果表",
        "log_dir": "04_实验结果/运行日志",
    },
}


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``."""
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path) -> dict:
    """Load YAML, fill defaults, validate, and return the config dict."""
    path = Path(path)
    user_config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    config = deep_merge(DEFAULT_CONFIG, user_config)
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    """Fail fast on configuration mistakes that would invalidate a run."""
    dataset = config.get("dataset", {})
    dataset_name = dataset.get("name")
    if dataset_name not in ALLOWED_DATASETS:
        raise ValueError(
            f"dataset.name must be one of {sorted(ALLOWED_DATASETS)}, got {dataset_name!r}"
        )

    arms = config.get("arms", [])
    if not arms:
        raise ValueError("arms must not be empty")
    unknown_arms = sorted(set(arms) - ALLOWED_ARMS)
    if unknown_arms:
        raise ValueError(f"unknown arms: {unknown_arms}; allowed: {sorted(ALLOWED_ARMS)}")

    seeds = config.get("evaluation", {}).get("seeds", [])
    if not seeds:
        raise ValueError("evaluation.seeds must not be empty")
    if any(int(seed) != seed for seed in seeds):
        raise ValueError(f"evaluation.seeds must be integers, got {seeds!r}")

    density = float(config.get("equiv_dropout", {}).get("density", 0.5))
    if not 0.0 < density <= 1.0:
        raise ValueError(f"equiv_dropout.density must be in (0, 1], got {density}")

    n_components = int(config.get("preprocess", {}).get("csp_components", 6))
    if n_components <= 0:
        raise ValueError(f"preprocess.csp_components must be > 0, got {n_components}")

    if dataset_name == "synthetic":
        for key in ("n_subjects", "n_sessions", "n_runs", "n_classes", "n_trials_per_class"):
            if int(dataset.get(key, 0)) <= 0:
                raise ValueError(f"dataset.{key} must be > 0 for synthetic data")
    else:
        subjects = dataset.get("subjects", [])
        if not subjects:
            raise ValueError("dataset.subjects must not be empty for bnci2014_001")


def config_hash(config: dict) -> str:
    """Stable hash of the effective config for the run log."""
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def save_config(config: dict, path: str | Path) -> Path:
    """Write a config snapshot as YAML."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path
