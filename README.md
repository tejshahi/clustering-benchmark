# clustering-benchmark

Benchmarking clustering algorithms on Sentinel-2 satellite time-series crop
datasets, with a focus on separating **perennial vs annual** crops.

## Data

Two datasets sit alongside this code (already present, not tracked in git —
see `.gitignore`):

| Dataset | Samples | Bands | Timesteps | dtype | Size |
|---|---|---|---|---|---|
| `S2Agri-10pc-17` | 5,850,881 | 10 | 24 | uint16 | ~2.8 GB |
| `TimeSen2Crop` | 1,135,511 | 9 | 365 | float32 | ~14.9 GB |

Each dataset has `_X.npy` (samples x bands x timesteps) and `_y.npy` (integer
class labels). `TimeSen2Crop` additionally has `_meta.npy`.

**TODO:** the raw integer labels in `_y.npy` are crop-type codes, not a
perennial/annual flag. Fill in `config/default.yaml`'s `label_groups` once
the label legend for each dataset is known, so `metrics.py` can score
clusters against the perennial/annual distinction directly instead of the
full multi-class labels.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Note:** this folder lives in OneDrive. Creating the venv *inside* it
> works but file-heavy operations (installing packages, deleting the venv)
> are much slower over the sync layer than on local disk — installing the
> full dependency list took ~45s outside OneDrive vs. timing out repeatedly
> inside it. If installs seem to hang, create the venv elsewhere instead,
> e.g. `python3 -m venv ~/.venvs/clustering-benchmark`, and just point your
> shell/IDE at that interpreter when working in this project.

## Project layout

```
config/default.yaml          # dataset paths, sampling, feature & clustering settings
src/cluster_bench/
  data.py                     # memmapped loading + stratified subsampling
  preprocessing.py            # feature extraction (flatten / per-band stats) + scaling
  clustering.py                # uniform wrappers: KMeans, MiniBatchKMeans, Agglomerative, GMM, DBSCAN
  metrics.py                   # ARI, NMI, homogeneity/completeness/v-measure, silhouette
  viz.py                        # PCA scatter + metric comparison plots
  run_benchmark.py             # CLI entry point
scripts/run_benchmark.py       # thin wrapper to run without `python -m`
notebooks/                     # exploratory analysis
results/<dataset>/             # per-run outputs: summary.csv, run_config.json, plots
tests/                         # sanity tests (memmap-based, don't load full arrays)
```

## Running

Both datasets are millions of rows, so the runner works on a **stratified
subsample** (default: 20,000 rows, configurable in `config/default.yaml` or
via `--n-samples`) and extracts compact per-band statistics rather than
clustering on raw flattened time series.

```bash
python -m cluster_bench.run_benchmark --dataset s2agri
python -m cluster_bench.run_benchmark --dataset timesen2crop --n-samples 5000
```

By default `n_clusters` is auto-detected as the number of distinct class
labels in the dataset's `y` (see `data.count_unique_labels`). Override it
with `--n-clusters N` on the CLI, or set `clustering.n_clusters` in
`config/default.yaml` (e.g. to `2`, once you want clusters to target the
perennial/annual split specifically rather than the full label set).

Or, without installing the package:

```bash
python scripts/run_benchmark.py --dataset s2agri
```

Each run writes to `results/<dataset>/`:
- `summary.csv` — metrics per algorithm, sorted by Adjusted Rand Index
- `run_config.json` — the exact settings used
- `scatter_<algorithm>.png` — PCA scatter, true label vs predicted cluster
- `ari_comparison.png` — bar chart comparing algorithms

## Tests

```bash
pip install pytest
pytest
```

## Next steps

- Fill in `label_groups` in `config/default.yaml` once the perennial/annual
  mapping for each dataset's class codes is known.
- Consider `tslearn` (DTW-based `TimeSeriesKMeans`, `KShape`) for clustering
  directly on the raw time-series shape rather than summary statistics —
  install separately (`pip install tslearn`), it's commented out in
  `requirements.txt` since it can be slow to build on some platforms.
- `TimeSen2Crop`'s `_meta.npy` is unused so far — inspect what it encodes
  (region/tile id? acquisition quality flag?) and fold it in if useful.
