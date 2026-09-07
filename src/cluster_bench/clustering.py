"""Thin, uniform wrappers around sklearn clustering algorithms so the
benchmark runner can loop over them generically.

Each factory takes n_clusters + seed and any algorithm-specific kwargs, and
returns an unfitted estimator implementing fit_predict.
"""
from __future__ import annotations

from typing import Callable

from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans, MiniBatchKMeans
from sklearn.mixture import GaussianMixture


def _kmeans(n_clusters: int, seed: int, **kwargs):
    return KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto", **kwargs)


def _minibatch_kmeans(n_clusters: int, seed: int, **kwargs):
    return MiniBatchKMeans(n_clusters=n_clusters, random_state=seed, n_init="auto", **kwargs)


def _agglomerative(n_clusters: int, seed: int, **kwargs):
    # Agglomerative clustering has no random_state (deterministic).
    return AgglomerativeClustering(n_clusters=n_clusters, **kwargs)


def _gmm(n_clusters: int, seed: int, **kwargs):
    return GaussianMixture(n_components=n_clusters, random_state=seed, **kwargs)


def _dbscan(n_clusters: int, seed: int, **kwargs):
    # DBSCAN doesn't take n_clusters directly — it infers cluster count from
    # eps/min_samples. n_clusters is accepted (and ignored) so it fits the
    # same factory signature as the other algorithms.
    return DBSCAN(**kwargs)


ALGORITHMS: dict[str, Callable] = {
    "kmeans": _kmeans,
    "minibatch_kmeans": _minibatch_kmeans,
    "agglomerative": _agglomerative,
    "gmm": _gmm,
    "dbscan": _dbscan,
}


def build_algorithm(name: str, n_clusters: int, seed: int = 42, **kwargs):
    if name not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm '{name}'. Options: {list(ALGORITHMS)}")
    return ALGORITHMS[name](n_clusters=n_clusters, seed=seed, **kwargs)


def fit_predict(name: str, X, n_clusters: int, seed: int = 42, **kwargs):
    model = build_algorithm(name, n_clusters=n_clusters, seed=seed, **kwargs)
    labels = model.fit_predict(X)
    return model, labels
