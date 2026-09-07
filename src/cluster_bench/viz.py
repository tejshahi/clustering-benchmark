"""Plotting helpers for the benchmark: PCA scatter of clusters vs true
labels, and a metric comparison bar chart across algorithms.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def plot_cluster_scatter(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, title: str, out_path: Path):
    """2D PCA projection, side-by-side: colored by true label vs by predicted cluster."""
    X2 = PCA(n_components=2, random_state=0).fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, labels, subtitle in [
        (axes[0], y_true, "true label"),
        (axes[1], y_pred, "predicted cluster"),
    ]:
        sc = ax.scatter(X2[:, 0], X2[:, 1], c=labels, cmap="tab20", s=4, alpha=0.6)
        ax.set_title(subtitle)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_metric_comparison(summary_df, metric: str, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(summary_df["algorithm"], summary_df[metric])
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
