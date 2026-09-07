"""Cluster evaluation metrics.

Two kinds are reported:
  - Internal (no ground truth needed): silhouette score.
  - External (needs the dataset's y labels): ARI, NMI, homogeneity,
    completeness, v-measure. These treat y as ground truth even though the
    raw labels are crop-type codes, not a perennial/annual flag — see
    config/default.yaml's `label_groups` for mapping codes to that
    coarser distinction once it's known.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import metrics as skm


def evaluate_clustering(X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    result = {
        "n_samples": X.shape[0],
        "n_clusters_found": len(set(y_pred.tolist()) - {-1}),  # -1 = DBSCAN noise
        "noise_fraction": float(np.mean(y_pred == -1)) if -1 in y_pred else 0.0,
        "adjusted_rand_index": skm.adjusted_rand_score(y_true, y_pred),
        "normalized_mutual_info": skm.normalized_mutual_info_score(y_true, y_pred),
        "homogeneity": skm.homogeneity_score(y_true, y_pred),
        "completeness": skm.completeness_score(y_true, y_pred),
        "v_measure": skm.v_measure_score(y_true, y_pred),
    }

    # Silhouette needs >= 2 clusters and is expensive; subsample if huge.
    n_unique = len(set(y_pred.tolist()))
    if 1 < n_unique < X.shape[0]:
        sample_size = min(5000, X.shape[0])
        result["silhouette"] = skm.silhouette_score(X, y_pred, sample_size=sample_size, random_state=0)
    else:
        result["silhouette"] = float("nan")

    return result


def summarize(results: list[dict]) -> pd.DataFrame:
    """results: list of {"algorithm": str, **evaluate_clustering(...)}"""
    df = pd.DataFrame(results)
    cols = [c for c in [
        "algorithm", "n_clusters_found", "noise_fraction",
        "adjusted_rand_index", "normalized_mutual_info",
        "homogeneity", "completeness", "v_measure", "silhouette",
    ] if c in df.columns]
    return df[cols].sort_values("adjusted_rand_index", ascending=False).reset_index(drop=True)
