"""Shared utilities for the fruit-fly brain EEG project.

Import from the submodules directly, e.g.::

    from common_utils.splits import make_loso_splits
    from common_utils.leakage_checks import assert_subject_disjoint
"""

from .config import DEFAULT_CONFIG, config_hash, load_config, validate_config
from .leakage_checks import (
    CanaryResult,
    LeakageError,
    assert_all_subject_sessions_in_test,
    assert_no_duplicate_epochs,
    assert_no_test_subject_in_fit,
    assert_subject_disjoint,
    assert_trial_grouped,
    label_permutation_canary,
    run_fold_checks,
)
from .metadata import (
    REQUIRED_METADATA_COLUMNS,
    TRIAL_KEY,
    metadata_summary,
    normalize_metadata,
    validate_metadata,
)
from .seeding import DEFAULT_SEEDS, derive_seed, set_global_seed
from .splits import (
    Fold,
    folds_to_records,
    load_folds,
    make_loso_splits,
    make_nested_loso_splits,
    save_folds,
)

__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_SEEDS",
    "CanaryResult",
    "Fold",
    "LeakageError",
    "REQUIRED_METADATA_COLUMNS",
    "TRIAL_KEY",
    "assert_all_subject_sessions_in_test",
    "assert_no_duplicate_epochs",
    "assert_no_test_subject_in_fit",
    "assert_subject_disjoint",
    "assert_trial_grouped",
    "config_hash",
    "derive_seed",
    "folds_to_records",
    "label_permutation_canary",
    "load_config",
    "load_folds",
    "make_loso_splits",
    "make_nested_loso_splits",
    "metadata_summary",
    "normalize_metadata",
    "run_fold_checks",
    "save_folds",
    "set_global_seed",
    "validate_config",
    "validate_metadata",
]
