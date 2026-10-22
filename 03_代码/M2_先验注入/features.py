"""Stage 1 feature alignment: multi-band CSP to d_in = 32.

The draft's Stage 1 extracts a 32-dimensional "brain-region" feature vector
before the masked fly-topology layer. We use four motor-imagery bands and 8
CSP components per band (4 x 8 = 32). All filtering is fixed and label-free;
the CSP filters are fitted **inside each LOSO training fold** by the sklearn
Pipeline in ``evaluate.py``.
"""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from mne.decoding import CSP
from scipy.signal import butter, sosfiltfilt
from sklearn.base import BaseEstimator, TransformerMixin

DEFAULT_BANDS = ((8.0, 12.0), (12.0, 16.0), (16.0, 24.0), (24.0, 32.0))
DEFAULT_COMPONENTS = 8


def bandpass_epochs(
    X: np.ndarray,
    sfreq: float,
    low: float,
    high: float,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth band-pass along the time axis.

    ``X`` has shape ``(n_epochs, n_channels, n_times)``. The filter is fixed and
    data-independent; it may be applied before splitting without leakage.
    """
    X = np.asarray(X, dtype=np.float64)
    if not 0 < low < high < sfreq / 2.0:
        raise ValueError(f"invalid band ({low}, {high}) for sfreq={sfreq}")
    sos = butter(order, [low, high], btype="bandpass", fs=sfreq, output="sos")
    filtered = sosfiltfilt(sos, X, axis=-1)
    return filtered.astype(np.float32, copy=False)


def _fit_band(
    X: np.ndarray,
    y,
    low: float,
    high: float,
    sfreq: float,
    n_components: int,
    reg,
    log: bool,
):
    Xb = bandpass_epochs(X, sfreq, low, high)
    csp = CSP(
        n_components=int(n_components),
        reg=reg,
        log=bool(log),
        norm_trace=False,
    )
    csp.fit(Xb, y)
    return csp, csp.transform(Xb).astype(np.float32, copy=False)


def _transform_band(
    X: np.ndarray,
    csp: CSP,
    low: float,
    high: float,
    sfreq: float,
) -> np.ndarray:
    Xb = bandpass_epochs(X, sfreq, low, high)
    return csp.transform(Xb).astype(np.float32, copy=False)


class MultiBandCSP(BaseEstimator, TransformerMixin):
    """Concatenate CSP features from several frequency bands."""

    def __init__(
        self,
        bands=DEFAULT_BANDS,
        n_components: int = DEFAULT_COMPONENTS,
        sfreq: float = 125.0,
        reg=None,
        log: bool = True,
        n_jobs: int = 1,
    ):
        self.bands = bands
        self.n_components = n_components
        self.sfreq = sfreq
        self.reg = reg
        self.log = log
        self.n_jobs = n_jobs

    def _make_csp(self) -> CSP:
        return CSP(
            n_components=int(self.n_components),
            reg=self.reg,
            log=bool(self.log),
            norm_trace=False,
        )

    def _validate(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError(f"expected (epochs, channels, times), got {X.shape}")
        if X.shape[1] < int(self.n_components):
            raise ValueError(
                f"n_components={self.n_components} exceeds n_channels={X.shape[1]}"
            )
        return X

    def _jobs(self) -> int:
        return 1 if self.n_jobs in (None, 0) else int(self.n_jobs)

    def fit(self, X, y):
        self.fit_transform(X, y)
        return self

    def transform(self, X):
        X = self._validate(X)
        if not hasattr(self, "csps_"):
            raise RuntimeError("MultiBandCSP must be fitted before transform")
        jobs = self._jobs()
        if jobs == 1:
            parts = [
                _transform_band(X, csp, float(low), float(high), float(self.sfreq))
                for (low, high), csp in zip(self.bands, self.csps_)
            ]
        else:
            parts = Parallel(n_jobs=jobs, max_nbytes=None)(
                delayed(_transform_band)(
                    X, csp, float(low), float(high), float(self.sfreq)
                )
                for (low, high), csp in zip(self.bands, self.csps_)
            )
        return np.hstack(parts).astype(np.float32, copy=False)

    def fit_transform(self, X, y=None, **fit_params):
        """Fit and transform with one filtering pass per band."""
        X = self._validate(X)
        jobs = self._jobs()
        if jobs == 1:
            results = [
                _fit_band(
                    X,
                    y,
                    float(low),
                    float(high),
                    float(self.sfreq),
                    int(self.n_components),
                    self.reg,
                    bool(self.log),
                )
                for low, high in self.bands
            ]
        else:
            results = Parallel(n_jobs=jobs, max_nbytes=None)(
                delayed(_fit_band)(
                    X,
                    y,
                    float(low),
                    float(high),
                    float(self.sfreq),
                    int(self.n_components),
                    self.reg,
                    bool(self.log),
                )
                for low, high in self.bands
            )
        self.csps_ = [item[0] for item in results]
        self.n_features_out_ = len(self.bands) * int(self.n_components)
        return np.hstack([item[1] for item in results]).astype(np.float32, copy=False)
