"""Turn raw (n_samples, n_bands, n_timesteps) time-series arrays into
feature vectors suitable for classic clustering algorithms.
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler


def flatten_features(X: np.ndarray) -> np.ndarray:
    """(n_samples, n_bands, n_timesteps) -> (n_samples, n_bands * n_timesteps)."""
    n = X.shape[0]
    return X.reshape(n, -1).astype(np.float64)


def band_stats_features(X: np.ndarray) -> np.ndarray:
    """Per-band summary statistics over the time axis: a compact,
    phenology-agnostic feature set (mean, std, min, max, and the
    10th/50th/90th percentiles) that's much cheaper than flattening full
    time series and tends to work well for annual-vs-perennial separation,
    since it captures the *shape* of the seasonal signal rather than exact
    timing.

    Input:  (n_samples, n_bands, n_timesteps)
    Output: (n_samples, n_bands * 7)
    """
    Xf = X.astype(np.float64)
    stats = [
        Xf.mean(axis=2),
        Xf.std(axis=2),
        Xf.min(axis=2),
        Xf.max(axis=2),
        np.percentile(Xf, 10, axis=2),
        np.percentile(Xf, 50, axis=2),
        np.percentile(Xf, 90, axis=2),
    ]
    return np.concatenate(stats, axis=1)


FEATURE_EXTRACTORS = {
    "flatten": flatten_features,
    "band_stats": band_stats_features,
}


def extract_features(X: np.ndarray, mode: str = "band_stats") -> np.ndarray:
    if mode not in FEATURE_EXTRACTORS:
        raise ValueError(f"Unknown feature_mode '{mode}'. Options: {list(FEATURE_EXTRACTORS)}")
    return FEATURE_EXTRACTORS[mode](X)


def scale_features(X: np.ndarray, scaler: StandardScaler | None = None):
    """Standardize features (zero mean, unit variance) column-wise.

    Returns (X_scaled, fitted_scaler) so the same scaler can be reused
    (e.g. fit on a train split, applied to a test split).
    """
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    return X_scaled, scaler
