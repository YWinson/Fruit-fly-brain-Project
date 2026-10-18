"""M1 arm registry.

``baseline`` is CSP + LDA. ``equiv_dropout`` is the same pipeline plus a fixed
random feature mask at a configurable density. The mask is the M1 proxy for
the equivalent-dropout control: it matches the sparsity of a future structural
mask without using any biological structure. It is not literal training-time
dropout; when a neural arm is added (EEGNet / M5), the same arm interface can
host a real dropout module.
"""

from __future__ import annotations

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline

from common_utils.seeding import derive_seed

from .preprocess import RandomFeatureMask, build_preprocessor

ARM_NAMES = ("baseline", "equiv_dropout")


def make_classifier(config: dict):
    """Build the M1 classifier."""
    model = config.get("model", {})
    model_type = model.get("type", "lda")
    if model_type != "lda":
        raise ValueError(f"unsupported model.type: {model_type!r} (M1 supports 'lda')")
    return LinearDiscriminantAnalysis(
        solver=model.get("lda_solver", "lsqr"),
        shrinkage=model.get("lda_shrinkage", "auto"),
    )


def arm_model_name(arm: str) -> str:
    """Human-readable model name for the IF-2 table."""
    names = {
        "baseline": "csp_lda",
        "equiv_dropout": "csp_lda_random_mask",
    }
    if arm not in names:
        raise ValueError(f"unknown arm: {arm!r}; allowed: {ARM_NAMES}")
    return names[arm]


def arm_mask_id(arm: str, config: dict, seed: int, fold_id: int) -> str:
    """Describe the arm-specific mask for the IF-2 table."""
    if arm != "equiv_dropout":
        return ""
    density = float(config["equiv_dropout"]["density"])
    mask_seed = derive_seed(seed, fold_id, "equiv_dropout")
    return f"random_mask_density={density:g}_seed={mask_seed}"


def build_arm(arm: str, config: dict, seed: int, fold_id: int) -> Pipeline:
    """Build a fresh, unfitted pipeline for one arm/fold/seed."""
    if arm not in ARM_NAMES:
        raise ValueError(f"unknown arm: {arm!r}; allowed: {ARM_NAMES}")

    preprocessor = build_preprocessor(config)
    steps = list(preprocessor.steps)

    if arm == "equiv_dropout":
        mask_seed = derive_seed(seed, fold_id, "equiv_dropout")
        steps.append(
            (
                "mask",
                RandomFeatureMask(
                    density=float(config["equiv_dropout"]["density"]),
                    seed=mask_seed,
                ),
            )
        )

    steps.append(("clf", make_classifier(config)))
    return Pipeline(steps)
