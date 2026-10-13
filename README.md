# Relational Fraud Detection — IEEE-CIS

**A reproducible raw-data → causal transaction graph → ML benchmark project.**

We tested whether message passing over inferred card/address/device relations
improves future-period fraud detection over transaction-only models. We built
our own temporal graph from the IEEE-CIS Fraud Detection dataset rather than
starting with a prebuilt academic graph.

> **Observed conclusion:** Graph design matters, but our single-layer GraphSAGE
> variants did **not** beat XGBoost. This project documents negative results
> rather than selectively reporting only the strongest GNN run.

## Original exploratory results

| Model | Graph | Edges | Val AP (PR-AUC) | Test AP | Test ROC-AUC |
|---|---|---:|---:|---:|---:|
| MLP (first run) | none | — | 0.4785 | 0.4542 | 0.8644 |
| MLP (rerun) | none | — | 0.4930 | 0.4061 | 0.8432 |
| GraphSAGE | card1 + addr1 | 6,393,246 | 0.4838 | 0.4538 | 0.8599 |
| GraphSAGE | card1 + DeviceInfo | 970,760 | 0.4994 | 0.4657 | 0.8360 |
| GraphSAGE | combined unique edges | 7,266,146 | 0.4939 | 0.4670 | 0.8612 |
| **XGBoost** | none | — | **0.6723** | **0.5407** | **0.8918** |

These results came from earlier interactive notebook runs, **not from executing
the packaged code in this archive**. Training is stochastic and computationally
expensive. Results may vary with seeds/library versions. Neural-model runs
should be repeated across seeds for defensible uncertainty estimates. See
[`docs/methodology.md`](docs/methodology.md) for leakage/test-selection caveats.

## Repository structure

```text
relational-fraud-detection/
├── README.md
├── pyproject.toml
├── .python-version
├── .gitignore
├── src/fraud_graph/
│   ├── config.py              # Frozen training and graph settings
│   ├── paths.py               # One canonical project root
│   ├── graph.py               # Historical edge construction/validation
│   ├── preprocessing.py       # Train-only robust numerical/category pipelines
│   ├── data.py                # Memmapped arrays, Dataset, NeighborLoader
│   ├── models.py              # NodeEncoder, MLP, GraphSAGE
│   ├── training.py            # BCE, optimization, AP/ROC-AUC, early stopping
│   └── utils.py               # Seeds, JSON outputs
├── scripts/
│   ├── download_data.py
│   ├── build_dataset.py
│   ├── build_graphs.py
│   ├── migrate_legacy_artifacts.py
│   ├── train_mlp.py
│   ├── train_gnn.py
│   ├── train_xgboost.py
│   └── compare_results.py
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_graph_design.ipynb
│   ├── 03_baselines.ipynb
│   └── 04_gnn_experiments.ipynb
├── tests/
│   ├── test_graph.py
│   ├── test_preprocessing.py
│   └── test_models.py
├── docs/
│   ├── methodology.md
│   └── reference_results.csv
├── data/{raw,interim,processed}/
└── artifacts/{models,metrics,figures}/
```

The `data/` and `artifacts/` directories are intentionally ignored except
for `.gitkeep`; **no raw Kaggle data, PII, feature arrays, or large model files
are included in Git**.

## Setup

Use Python 3.11 and [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync --group dev
uv run pytest -q
```

PyTorch + PyG binary extension installations can be platform-specific. For
PyG `NeighborLoader`, install a matching wheel of **`pyg-lib`** or
`torch-sparse`, according to your active Torch/CUDA version, as described at
[PyG installation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).
Example matching our earlier Lightning Studio build (`torch 2.14.0+cu130`):

```bash
uv pip install --python .venv/bin/python pyg-lib \
  -f 'https://data.pyg.org/whl/torch-2.14.0+cu130.html'
```

Only use this exact wheel URL if your **actual environment** has that Torch
build. Verify your notebook's `sys.executable`: the kernel and `uv` must target
the same environment. On low-RAM machines, run data preparation separately
from model training and restart the kernel/process between stages.

## Obtain the raw dataset

Go to [Kaggle IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection),
accept the competition rules, and authenticate the Kaggle CLI. Only training
files are needed. Run:

```bash
uv run python scripts/download_data.py
```

Or manually put `train_transaction.csv` and `train_identity.csv` into
`data/raw/`. The downloader extracts each `.csv.zip`, checks filenames,
and skips files already present.

## Full reproduction (from the repo root)

```bash
# 1. Convert raw data into ordered matrices, labels, masks, metadata
uv run python scripts/build_dataset.py

# 2. Build both causal relations and their deduplicated union
uv run python scripts/build_graphs.py --relation all --window-days 30 --k 20

# 3. Train node-only MLP
uv run python scripts/train_mlp.py --seed 42

# 4. Train each 1-layer GraphSAGE independently
uv run python scripts/train_gnn.py --graph address --seed 42
uv run python scripts/train_gnn.py --graph device --seed 42
uv run python scripts/train_gnn.py --graph combined --seed 42

# 5. Train weighted native-category XGBoost
uv run python scripts/train_xgboost.py --rounds 5000 --patience 200 --seed 42

# 6. Print newly run experiment scores
uv run python scripts/compare_results.py
```

Raw data preprocessing uses sizeable RAM (the raw tables have ~590k rows and
hundreds of columns). The numerical matrix is written to a `.npy` memory map
**one column at a time**, avoiding the repeated full DataFrame copies that
crashed our original Jupyter kernel. Training reads memmaps and uses CPU/GPU
based on availability. GraphSAGE training requires a working compiled PyG
sampler (`pyg-lib` or `torch-sparse`).

### Already have the original notebook artifacts?

You don't need to rerun the expensive extraction. From the repository root:

```bash
uv run python scripts/migrate_legacy_artifacts.py \
  --source /path/to/old/data/processed --copy
```

This imports the previous `numeric_features.npy`,
`categorical_features.npy`, masks, targets, saved graphs, and converts the
trusted `preprocessing.pkl` into our versioned JSON metadata. Remove `--copy`
to use symlinks (Linux/macOS) and save disk. Provide the raw training CSV if
available to recover the exact node ordering and times for graph rebuilding.

## Why graphs are causal

For each destination transaction at time `t`, incoming nodes must have the
same composite key (`card1` + `addr1`, or `card1` + `DeviceInfo`), earlier
`TransactionDT`, occur in `[t - 30 days, t)`, and be among the **20** most
recent qualifying predecessors. Timestamp ties do not create links. The union
may have up to **40** incoming edges per transaction because each individual
relation has its own 20-neighbor cap.

The timestamp is relative to an unspecified dataset start. The date-ordered
70/15/15 split in the original run produced:

| Split | Rows | Fraud fraction |
|---|---:|---:|
| Train | 413,378 | 0.03517 |
| Validation | 88,581 | 0.03434 |
| Test | 88,581 | 0.03480 |

The shared `NodeEncoder` uses 397 clipped/imputed/standardized numeric inputs
and 37 categorical ID fields with distinct embeddings. Both MLP and GraphSAGE
use weighted `BCEWithLogitsLoss`, and we select their epoch by **validation**
sklearn Average Precision. On sampled PyG batches only **seed node labels**
receive loss; historical context nodes supply feature messages only.

**Methodology note:** Dataset-wide EDA and earlier test results influenced some
later exploratory graph hypotheses. This is therefore not a publication-quality
sealed holdout experiment. Future improvements should be judged on a newly
reserved time period. See [methodology](docs/methodology.md).

## Next research directions (not implemented here)

- XGBoost with point-in-time historical transaction-count and recency features
- Heterogeneous transaction ↔ card/device/entity graph, with relation-aware GNN
- Compare a count-sensitive GIN/sum aggregation against GraphSAGE mean
- Multi-seed uncertainty, cost-aware precision@K, and prospective backtesting

These are **future ideas**, not features claimed to be implemented in this repo.
