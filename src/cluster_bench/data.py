"""Loading and subsampling for the benchmark's Sentinel-2 datasets.

Both datasets are large (millions of samples), so X arrays are opened with
mmap_mode="r" by default: nothing is read into memory until it's indexed,
which makes stratified subsampling cheap even on the 14GB TimeSen2Crop file.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


@dataclass
class Dataset:
    key: str
    name: str
    X: np.ndarray  # (n_samples, n_bands, n_timesteps), memmapped
    y: np.ndarray  # (n_samples,)
    meta: Optional[np.ndarray]
    description: str


def load_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_dataset(key: str, config: Optional[dict] = None, mmap: bool = True) -> Dataset:
    """Load one dataset by its config key ("s2agri" or "timesen2crop").

    Arrays are memmapped by default (mmap=True) so this is cheap even for
    the multi-GB files; use subsample() afterwards to pull a manageable,
    in-memory slice for clustering.
    """
    config = config or load_config()
    spec = config["datasets"][key]
    mmap_mode = "r" if mmap else None

    x_path = PROJECT_ROOT / spec["x_path"]
    y_path = PROJECT_ROOT / spec["y_path"]
    X = np.load(x_path, mmap_mode=mmap_mode)
    y = np.load(y_path, mmap_mode=mmap_mode)

    meta = None
    if spec.get("meta_path"):
        meta = np.load(PROJECT_ROOT / spec["meta_path"], mmap_mode=mmap_mode)

    return Dataset(
        key=key,
        name=spec["name"],
        X=X,
        y=y,
        meta=meta,
        description=spec.get("description", ""),
    )


def subsample(
    ds: Dataset,
    n_samples: int,
    stratify_by_label: bool = True,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Pull an in-memory (X, y) subsample of size <= n_samples.

    If stratify_by_label, samples are drawn proportionally from each class
    in ds.y so rare classes aren't dropped entirely. Returns dense (not
    memmapped) arrays, safe to feed straight into sklearn.
    """
    rng = np.random.default_rng(seed)
    n_total = ds.y.shape[0]
    n_samples = min(n_samples, n_total)

    if not stratify_by_label:
        idx = rng.choice(n_total, size=n_samples, replace=False)
        idx.sort()
        return np.asarray(ds.X[idx]), np.asarray(ds.y[idx])

    labels, counts = np.unique(ds.y, return_counts=True)
    proportions = counts / counts.sum()
    per_class = np.maximum(1, np.round(proportions * n_samples).astype(int))

    chosen = []
    # y is potentially memmapped; np.where over it materializes indices only,
    # which is fine at this size (millions of ints, not the full X array).
    y_full = np.asarray(ds.y)
    for label, n_take in zip(labels, per_class):
        class_idx = np.flatnonzero(y_full == label)
        n_take = min(n_take, class_idx.shape[0])
        chosen.append(rng.choice(class_idx, size=n_take, replace=False))

    idx = np.concatenate(chosen)
    idx.sort()
    idx = idx[:n_samples]
    return np.asarray(ds.X[idx]), np.asarray(ds.y[idx])
