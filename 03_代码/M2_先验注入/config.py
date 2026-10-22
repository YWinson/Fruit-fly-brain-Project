"""M2 configuration defaults and validation."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from common_utils.config import deep_merge

ALLOWED_ARMS = (
    "bio_sparse",
    "deg_rewired",
    "er_random",
    "equiv_dropout",
    "dense_baseline",
)
ALLOWED_GRANULARITIES = ("fine", "medium", "coarse")

M2_DEFAULT_CONFIG: dict = {
    "dataset": {
        "name": "bnci2014_001",
        "subjects": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "data_path": "02_数据/BNCI2014-001_原始",
        "n_classes": 4,
        "tmin": 0.0,
        "tmax": 2.0,
        "resample": 125,
        "fmin": 8.0,
        "fmax": 32.0,
        "sfreq": 125.0,
        "duration_s": 2.0,
    },
    "preprocess": {
        "bands": [[8, 12], [12, 16], [16, 24], [24, 32]],
        "csp_components": 8,
        "csp_reg": None,
        "csp_log": True,
        "csp_n_jobs": 1,
    },
    "model": {
        "hidden_dim": 64,
        "epochs": 100,
        "batch_size": 256,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "device": "cpu",
        "verbose": False,
    },
    "fly": {
        "raw_dir": "02_数据/fly_connectome/raw",
        "processed_dir": "02_数据/fly_connectome/processed",
        "granularity": "fine",
        "mask_density": 0.10,
        "mask_selection": "balanced",
        "min_neurons": 1,
        "superclasses": [
            "central",
            "sensory",
            "ascending",
            "descending",
            "motor",
            "sensory_ascending",
            "endocrine",
        ],
        "motif_null_samples": 1000,
    },
    "arms": [
        "bio_sparse",
        "deg_rewired",
        "er_random",
        "equiv_dropout",
        "dense_baseline",
    ],
    "evaluation": {
        "seeds": [0, 1, 2],
        "canary_tolerance": 0.10,
    },
    "output": {
        "results_dir": "04_实验结果/结果表",
        "log_dir": "04_实验结果/运行日志",
    },
}


def load_m2_config(path: str | Path) -> dict:
    path = Path(path)
    user = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    config = deep_merge(M2_DEFAULT_CONFIG, user)
    validate_m2_config(config)
    return config


def validate_m2_config(config: dict) -> None:
    dataset_name = config.get("dataset", {}).get("name")
    if dataset_name not in {"bnci2014_001", "synthetic"}:
        raise ValueError(f"unsupported dataset: {dataset_name!r}")

    arms = config.get("arms", [])
    unknown = sorted(set(arms) - set(ALLOWED_ARMS))
    if unknown:
        raise ValueError(f"unknown M2 arms: {unknown}; allowed: {ALLOWED_ARMS}")

    granularity = config.get("fly", {}).get("granularity", "fine")
    if granularity not in ALLOWED_GRANULARITIES:
        raise ValueError(
            f"fly.granularity must be one of {ALLOWED_GRANULARITIES}, got {granularity!r}"
        )

    density = float(config.get("fly", {}).get("mask_density", 0.10))
    if not 0.0 < density <= 1.0:
        raise ValueError(f"fly.mask_density must be in (0, 1], got {density}")

    selection = config.get("fly", {}).get("mask_selection", "balanced")
    if selection not in {"balanced", "global"}:
        raise ValueError(f"fly.mask_selection must be 'balanced' or 'global', got {selection!r}")

    pre = config.get("preprocess", {})
    bands = pre.get("bands", [])
    components = int(pre.get("csp_components", 8))
    if len(bands) * components != 32:
        raise ValueError(
            "Stage 1 must produce d_in=32 features: "
            f"{len(bands)} bands x {components} components != 32"
        )

    seeds = config.get("evaluation", {}).get("seeds", [])
    if not seeds:
        raise ValueError("evaluation.seeds must not be empty")


def copy_defaults() -> dict:
    return copy.deepcopy(M2_DEFAULT_CONFIG)
