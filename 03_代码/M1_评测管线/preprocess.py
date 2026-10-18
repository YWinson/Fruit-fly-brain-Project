"""Preprocessing transformers for M1.

Every transform that learns from data is a scikit-learn transformer. The M1
runner puts them in a ``Pipeline`` and fits that pipeline **inside each LOSO
fold**, so the held-out subject never contributes to a mean, covariance, CSP
filter, or feature mask.
"""

from __future__ import annotations

import numpy as np
from mne.decoding import CSP
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


class ChannelStandardScaler(BaseEstimator, TransformerMixin):
    """Per-channel standardisation fitted on training epochs only.

    Input shape: ``(n_epochs, n_channels, n_times)``.
    """

    def fit(self, X, y=None):
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError(f"expected 3-D epochs, got {X.shape}")
        # Accumulate in float64 for stable statistics, but keep the data and
        # the learned parameters in the input precision (normally float32).
        # This avoids a full float64 copy of the training epochs.
        self.mean_ = X.mean(axis=(0, 2), keepdims=True, dtype=np.float64).astype(
            X.dtype, copy=False
        )
        self.std_ = X.std(axis=(0, 2), keepdims=True, dtype=np.float64).astype(
            X.dtype, copy=False
        )
        self.std_[self.std_ < 1e-12] = 1.0
        return self

    def transform(self, X):
        X = np.asarray(X)
        return (X - self.mean_) / self.std_


class CSPTransformer(BaseEstimator, TransformerMixin):
    """Thin wrapper around :class:`mne.decoding.CSP`."""

    def __init__(self, n_components: int = 6, reg=None, log: bool = True):
        self.n_components = n_components
        self.reg = reg
        self.log = log

    def fit(self, X, y):
        self.csp_ = CSP(
            n_components=int(self.n_components),
            reg=self.reg,
            log=bool(self.log),
            norm_trace=False,
        )
        self.csp_.fit(X, y)
        return self

    def transform(self, X):
        return self.csp_.transform(X)


class RandomFeatureMask(BaseEstimator, TransformerMixin):
    """Fixed random feature mask — the M1 equivalent-sparsity control.

    The mask is generated once per fold from a derived seed. It does not use
    labels and does not look at test data. It is a *proxy* for the eventual
    equivalent-dropout arm: same feature count as a structural mask, but no
    biological structure.
    """

    def __init__(self, density: float = 0.5, seed: int = 0):
        self.density = density
        self.seed = seed

    def fit(self, X, y=None):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"expected 2-D features, got {X.shape}")
        if not 0.0 < float(self.density) <= 1.0:
            raise ValueError(f"density must be in (0, 1], got {self.density}")
        n_features = X.shape[1]
        n_keep = max(1, int(round(float(self.density) * n_features)))
        rng = np.random.default_rng(int(self.seed))
        selected = np.sort(rng.choice(n_features, size=n_keep, replace=False))
        self.mask_ = np.zeros(n_features, dtype=bool)
        self.mask_[selected] = True
        self.selected_features_ = selected
        return self

    def transform(self, X):
        X = np.asarray(X)
        return X[:, self.mask_]


def build_preprocessor(config: dict) -> Pipeline:
    """Build the shared scaler + CSP preprocessor."""
    pre = config["preprocess"]
    steps = []
    if pre.get("standardize", True):
        steps.append(("scaler", ChannelStandardScaler()))
    steps.append(
        (
            "csp",
            CSPTransformer(
                n_components=int(pre.get("csp_components", 6)),
                reg=pre.get("csp_reg"),
                log=bool(pre.get("csp_log", True)),
            ),
        )
    )
    return Pipeline(steps)
