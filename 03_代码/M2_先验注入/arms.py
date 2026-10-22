"""M2 arm registry: biological mask, degree-preserving null, ER null, dropout, dense.

Stage 1 (multi-band CSP) is fitted once per LOSO fold and shared by all arms;
only the mask/sign (and the dropout keep probability) differs. This is both
fairer and much faster than refitting four CSPs for every arm.
"""

from __future__ import annotations

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common_utils.seeding import derive_seed

from .clusters import GranularityResult
from .eegnet import EEGNetFeatureExtractor
from .features import MultiBandCSP
from .masks import (
    MaskError,
    bipartite_degree_preserving_rewire,
    build_bio_mask,
    dense_mask,
    er_random_mask,
    mask_stats,
)
from .model import MaskedMLPClassifier

ARM_NAMES = (
    "bio_sparse",
    "deg_rewired",
    "er_random",
    "equiv_dropout",
    "dense_baseline",
)

MODEL_NAMES = {
    "bio_sparse": "masked_mlp_bio",
    "deg_rewired": "masked_mlp_deg_rewired",
    "er_random": "masked_mlp_er_random",
    "equiv_dropout": "masked_mlp_dropout",
    "dense_baseline": "masked_mlp_dense",
}


def build_stage1(config: dict, seed: int = 0, fold_id: int = 0) -> Pipeline:
    """Stage 1: multi-band CSP or an EEGNet trunk + feature scaling.

    The stage is fitted once per LOSO fold and shared by every arm. For the
    EEGNet variant the trunk is trained on the training subjects with a
    temporary 4-class head; the head is discarded and the frozen 32-D features
    feed the masked M2 heads.
    """
    pre_cfg = config.get("preprocess", {})
    stage1_type = str(pre_cfg.get("stage1", "csp")).lower()

    if stage1_type == "eegnet":
        eeg = pre_cfg.get("eegnet", {})
        return Pipeline(
            [
                (
                    "eegnet",
                    EEGNetFeatureExtractor(
                        feature_dim=int(eeg.get("feature_dim", 32)),
                        F1=int(eeg.get("F1", 8)),
                        D=int(eeg.get("D", 2)),
                        F2=int(eeg.get("F2", 16)),
                        kernel_length=int(eeg.get("kernel_length", 64)),
                        dropout=float(eeg.get("dropout", 0.25)),
                        epochs=int(eeg.get("epochs", 100)),
                        batch_size=int(eeg.get("batch_size", 64)),
                        lr=float(eeg.get("lr", 1e-3)),
                        weight_decay=float(eeg.get("weight_decay", 1e-4)),
                        random_state=derive_seed(seed, fold_id, "eegnet_stage1"),
                        device=str(eeg.get("device", "cpu")),
                        verbose=bool(eeg.get("verbose", False)),
                        num_threads=int(eeg.get("num_threads", 4)),
                    ),
                ),
                ("scaler", StandardScaler()),
            ]
        )

    if stage1_type != "csp":
        raise ValueError(f"unknown preprocess.stage1: {stage1_type!r}")

    return Pipeline(
        [
            (
                "bands",
                MultiBandCSP(
                    bands=tuple(
                        tuple(band)
                        for band in pre_cfg.get(
                            "bands", ((8, 12), (12, 16), (16, 24), (24, 32))
                        )
                    ),
                    n_components=int(pre_cfg.get("csp_components", 8)),
                    sfreq=float(config["dataset"].get("resample", 125)),
                    reg=pre_cfg.get("csp_reg"),
                    log=bool(pre_cfg.get("csp_log", True)),
                    n_jobs=int(pre_cfg.get("csp_n_jobs", 1)),
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )


def _make_classifier(
    config: dict,
    mask: np.ndarray,
    sign: np.ndarray,
    seed: int,
    fold_id: int,
    dropout_keep_prob: float,
) -> MaskedMLPClassifier:
    model_cfg = config.get("model", {})
    return MaskedMLPClassifier(
        mask=mask,
        sign=sign,
        hidden_dim=int(model_cfg.get("hidden_dim", 64)),
        epochs=int(model_cfg.get("epochs", 200)),
        batch_size=int(model_cfg.get("batch_size", 128)),
        lr=float(model_cfg.get("lr", 1e-3)),
        weight_decay=float(model_cfg.get("weight_decay", 1e-4)),
        dropout_keep_prob=float(dropout_keep_prob),
        random_state=derive_seed(seed, fold_id, "mlp_init"),
        device=str(model_cfg.get("device", "cpu")),
        verbose=bool(model_cfg.get("verbose", False)),
    )


def build_mask_for_arm(
    arm: str,
    config: dict,
    seed: int,
    fold_id: int,
    granularity_result: GranularityResult,
) -> tuple[np.ndarray, np.ndarray, dict, float]:
    """Return ``(mask, sign, info, dropout_keep_prob)`` for one arm/fold/seed."""
    if arm not in ARM_NAMES:
        raise ValueError(f"unknown arm: {arm!r}; allowed: {ARM_NAMES}")

    fly_cfg = config.get("fly", {})
    density = float(fly_cfg.get("mask_density", 0.10))
    selection = str(fly_cfg.get("mask_selection", "balanced"))
    bio_mask, bio_sign, bio_info = build_bio_mask(
        granularity_result, density=density, selection=selection
    )

    if arm == "bio_sparse":
        mask, sign, info, keep = bio_mask, bio_sign, dict(bio_info), 1.0
    elif arm == "deg_rewired":
        if granularity_result.name == "coarse":
            raise MaskError(
                "deg_rewired is undefined for the coarse E=2/I=2 block mask: "
                "its degree sequence forces the block graph, so a degree-preserving "
                "null cannot move edges. Use er_random for the coarse boundary."
            )
        mask, rewire_info = bipartite_degree_preserving_rewire(
            bio_mask, seed=derive_seed(seed, fold_id, "deg_rewired")
        )
        sign = bio_sign
        info = dict(bio_info)
        info.update(rewire_info)
        info["mask_id"] = (
            f"deg_{granularity_result.name}_{selection}_p{density:g}"
            f"_seed{derive_seed(seed, fold_id, 'deg_rewired')}"
        )
        keep = 1.0
    elif arm == "er_random":
        mask = er_random_mask(
            64,
            32,
            n_edges=int(bio_mask.sum()),
            seed=derive_seed(seed, fold_id, "er_random"),
        )
        sign = np.ones_like(mask, dtype=np.int8)
        info = mask_stats(mask)
        info.update(
            {
                "granularity": granularity_result.name,
                "mask_id": (
                    f"er_{granularity_result.name}_p{density:g}"
                    f"_seed{derive_seed(seed, fold_id, 'er_random')}"
                ),
            }
        )
        keep = 1.0
    elif arm == "equiv_dropout":
        mask = dense_mask(64, 32)
        sign = np.ones_like(mask, dtype=np.int8)
        info = mask_stats(mask)
        info.update(
            {
                "granularity": granularity_result.name,
                "mask_id": f"equiv_dropout_keep{density:g}",
                "dropout_keep_prob": density,
            }
        )
        keep = density
    else:  # dense_baseline
        mask = dense_mask(64, 32)
        sign = np.ones_like(mask, dtype=np.int8)
        info = mask_stats(mask)
        info.update({"granularity": granularity_result.name, "mask_id": "dense"})
        keep = 1.0

    info["arm"] = arm
    info["model"] = MODEL_NAMES[arm]
    return mask, sign, info, keep


def build_classifier(
    arm: str,
    config: dict,
    seed: int,
    fold_id: int,
    granularity_result: GranularityResult,
) -> tuple[MaskedMLPClassifier, np.ndarray, np.ndarray, dict]:
    """Build only Stage 2/3 for one arm (Stage 1 features are shared)."""
    mask, sign, info, keep = build_mask_for_arm(
        arm, config, seed, fold_id, granularity_result
    )
    classifier = _make_classifier(
        config,
        mask=mask,
        sign=sign,
        seed=seed,
        fold_id=fold_id,
        dropout_keep_prob=keep,
    )
    return classifier, mask, sign, info


def build_arm(
    arm: str,
    config: dict,
    seed: int,
    fold_id: int,
    granularity_result: GranularityResult,
) -> tuple[Pipeline, np.ndarray, np.ndarray, dict]:
    """Build the full Stage 1 + Stage 2/3 pipeline (for single-arm debugging)."""
    classifier, mask, sign, info = build_classifier(
        arm, config, seed, fold_id, granularity_result
    )
    stage1 = build_stage1(config)
    pipeline = Pipeline([*stage1.steps, ("clf", classifier)])
    return pipeline, mask, sign, info
