"""Random-seed helpers used by every M1 arm."""

from __future__ import annotations

import hashlib
import random

import numpy as np

DEFAULT_SEEDS = list(range(10))


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and (if installed) PyTorch."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def derive_seed(base_seed: int, *parts: object) -> int:
    """Derive a stable 32-bit seed from a base seed and arbitrary parts.

    Used for fold-specific random masks: the mask is reproducible, but it is
    not the same mask in every fold.
    """
    payload = "|".join([str(base_seed), *[str(part) for part in parts]])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")
