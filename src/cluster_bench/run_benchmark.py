"""CLI entry point: run the clustering benchmark on one dataset.

Usage:
    python -m cluster_bench.run_benchmark --dataset s2agri
    python -m cluster_bench.run_benchmark --dataset timesen2crop --n-samples 5000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import data, preprocessing, clustering, metrics, viz


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark clustering algorithms on a crop time-series dataset.")
    p.add_argument("--dataset", required=True, choices=["s2agri", "timesen2crop"])
    p.add_argument("--config", default=str(data.DEFAULT_CONFIG_PATH))
    p.add_argument("--n-samples", type=int, default=None, help="Override config's sampling.n_samples")
    p.add_argument(
        "--n-clusters", type=int, default=None,
        help="Number of clusters. Default: auto — the number of distinct class labels "
             "in the dataset. Overrides config's clustering.n_clusters.",
    )
    p.add_argument("--algorithms", nargs="*", default=None, help="Override config's clustering.algorithms")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    config = data.load_config(args.config)
    seed = config.get("seed", 42)

    n_samples = args.n_samples or config["sampling"]["n_samples"]
    stratify = config["sampling"]["stratify_by_label"]
    algorithms = args.algorithms or config["clustering"]["algorithms"]
    feature_mode = config["preprocessing"]["feature_mode"]
    do_scale = config["preprocessing"]["scale"]

    out_dir = Path(config["output_dir"]) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{args.dataset}] loading (memmapped)...")
    ds = data.load_dataset(args.dataset, config=config)
    print(f"[{args.dataset}] full size: X={ds.X.shape} y={ds.y.shape}")

    # n_clusters precedence: --n-clusters CLI flag > config's clustering.n_clusters
    # (if set) > auto-detected from the number of distinct labels in y. Auto is the
    # default so clustering doesn't silently target the wrong number of groups when
    # a dataset's label count doesn't match whatever was last configured.
    config_n_clusters = config["clustering"].get("n_clusters")
    if args.n_clusters is not None:
        n_clusters = args.n_clusters
        n_clusters_source = "--n-clusters"
    elif config_n_clusters is not None:
        n_clusters = config_n_clusters
        n_clusters_source = "config"
    else:
        n_clusters = data.count_unique_labels(ds)
        n_clusters_source = "auto (unique labels in y)"
    print(f"[{args.dataset}] n_clusters={n_clusters} (source: {n_clusters_source})")

    print(f"[{args.dataset}] subsampling {n_samples} (stratified={stratify})...")
    X_raw, y = data.subsample(ds, n_samples=n_samples, stratify_by_label=stratify, seed=seed)

    print(f"[{args.dataset}] extracting features (mode={feature_mode})...")
    X_feat = preprocessing.extract_features(X_raw, mode=feature_mode)
    if do_scale:
        X_feat, _ = preprocessing.scale_features(X_feat)

    results = []
    for algo_name in algorithms:
        print(f"[{args.dataset}] running {algo_name}...")
        try:
            _, y_pred = clustering.fit_predict(algo_name, X_feat, n_clusters=n_clusters, seed=seed)
        except Exception as e:  # noqa: BLE001 - report and keep going
            print(f"  !! {algo_name} failed: {e}")
            continue
        scores = metrics.evaluate_clustering(X_feat, y, y_pred)
        scores["algorithm"] = algo_name
        results.append(scores)

        if not args.no_plots:
            viz.plot_cluster_scatter(
                X_feat, y, y_pred,
                title=f"{ds.name} — {algo_name}",
                out_path=out_dir / f"scatter_{algo_name}.png",
            )

    if not results:
        print("No algorithm produced results.")
        return

    summary = metrics.summarize(results)
    print("\n" + summary.to_string(index=False))

    summary.to_csv(out_dir / "summary.csv", index=False)
    with open(out_dir / "run_config.json", "w") as f:
        json.dump({
            "dataset": args.dataset, "n_samples": n_samples, "n_clusters": n_clusters,
            "algorithms": algorithms, "feature_mode": feature_mode, "scale": do_scale, "seed": seed,
        }, f, indent=2)

    if not args.no_plots:
        viz.plot_metric_comparison(
            summary, "adjusted_rand_index",
            title=f"{ds.name} — ARI by algorithm",
            out_path=out_dir / "ari_comparison.png",
        )

    print(f"\nResults written to {out_dir}/")


if __name__ == "__main__":
    main()
