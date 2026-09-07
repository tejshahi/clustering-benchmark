"""Lightweight tests: shape/consistency checks that don't require loading
full multi-GB arrays into memory (mmap keeps this fast).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cluster_bench import data, preprocessing


def test_load_config():
    config = data.load_config()
    assert "s2agri" in config["datasets"]
    assert "timesen2crop" in config["datasets"]


def test_load_and_subsample_s2agri():
    config = data.load_config()
    ds = data.load_dataset("s2agri", config=config)
    assert ds.X.shape[0] == ds.y.shape[0]
    X, y = data.subsample(ds, n_samples=200, seed=0)
    assert X.shape[0] == y.shape[0] <= 200
    assert X.shape[1:] == ds.X.shape[1:]


def test_feature_extraction_shapes():
    config = data.load_config()
    ds = data.load_dataset("s2agri", config=config)
    X, _ = data.subsample(ds, n_samples=50, seed=0)
    n_bands = X.shape[1]

    flat = preprocessing.extract_features(X, mode="flatten")
    assert flat.shape == (50, n_bands * X.shape[2])

    stats = preprocessing.extract_features(X, mode="band_stats")
    assert stats.shape == (50, n_bands * 7)
