# Relational Fraud Detection: When Do Graph Neural Networks Help?

**A reproducible study of temporal fraud detection on IEEE-CIS, comparing tabular ML, homogeneous GraphSAGE, and a causal heterogeneous GNN.**

> **Main finding:** On our chronological evaluation, **XGBoost was the strongest model** (test Average Precision / PR-AUC **0.5407**, ROC-AUC **0.8918**). Our best transaction-to-transaction GraphSAGE reached **0.4670** test AP, and a typed temporal heterogeneous GNN reached **0.4538**. The study illustrates that more edges and more complex architectures do not automatically provide better generalization.
>
> **Scope:** These are **our project's exploratory results on a holdout carved from the labeled IEEE-CIS training data**, not official Kaggle competition scores or a general claim about GNNs versus boosted trees.

## Research question

**Can relations between historical transactions improve future fraud detection beyond a strong transaction-only model?**

We investigate three complementary questions:

1. **Graph design:** Do plausible inferred relationships—shared card/address or card/device signatures—contain additional predictive signal?
2. **Architecture:** Does relation-aware message passing improve on homogeneous aggregation?
3. **Model selection:** Does that extra graph complexity justify itself against an XGBoost baseline using the same underlying transaction features?

Rather than beginning with a ready-made PyG benchmark, we build the problem from raw transaction and identity tables, define point-in-time neighborhoods, construct graph tensors, train models, and evaluate the outcome.

## Results

**Primary metric:** scikit-learn `average_precision_score` (AP; called **PR-AUC** throughout this repository). **Secondary metric:** ROC-AUC. Higher is better.

| Model | Graph / relational representation | Validation AP | Test AP | Test ROC-AUC |
|:--|:--|--:|--:|--:|
| MLP — original run | Transaction features only | 0.4785 | 0.4542 | 0.8644 |
| MLP — rerun | Transaction features only | 0.4930 | 0.4061 | 0.8432 |
| GraphSAGE V1 | `(card1, addr1)` | 0.4838 | 0.4538 | 0.8599 |
| GraphSAGE V1.1 | `(card1, DeviceInfo)` | **0.4994** | 0.4657 | 0.8360 |
| GraphSAGE V1.2 | Deduplicated union of both relations | 0.4939 | **0.4670** | 0.8612 |
| TemporalHeteroFraud V2 | Typed address/device history snapshots | 0.4849 | 0.4538 | 0.8445 |
| **XGBoost** | **Transaction features only** | **0.6723** | **0.5407** | **0.8918** |

**Observed effect:** The combined homogeneous GNN was **0.0737 AP below XGBoost** on test (`0.4670` versus `0.5407`). V2 did not improve on the homogeneous GNNs. The two MLP runs also showed meaningful run-to-run variation, so a single neural-network run should not be treated as a stable estimate.

Results were recorded during the original experiments. Re-running packaged scripts can produce different values due to seeds, installed versions, and training conditions. V1 historical results also appear in [`docs/reference_results.csv`](docs/reference_results.csv); V2 was recorded separately in `artifacts/metrics/hetero_v2_seed42.json` (ignored by Git) and is included in the table above.

### What the results do—and do not—say

- **Relation quality matters:** The device-based relation achieved better test AP than the card/address relation, despite using fewer edges.
- **More edges are not necessarily better:** Combining relations grew the graph substantially for only a small gain over device-only GraphSAGE.
- **A stronger architecture is not a guaranteed stronger predictor:** Our V2 preserves separate relation types and historical counts, but did not outperform boosted trees.
- **We have not disproved GNN-based fraud detection.** These results are specific to our inferred identities, feature availability, temporal split, architectures, and training procedure. See [Methodology and limitations](docs/methodology.md).

## Data and experimental design

**Dataset:** [IEEE-CIS Fraud Detection (Kaggle)](https://www.kaggle.com/competitions/ieee-fraud-detection), combining `train_transaction.csv` with the optional `train_identity.csv` by unique `TransactionID`.

| Item | Original experiment |
|:--|--:|
| Transactions / transaction nodes | 590,540 |
| Fraud prevalence | ~3.50% |
| Numeric node features | 397 |
| Categorical feature fields | 37 |
| Chronological train / validation / test | 413,378 / 88,581 / 88,581 |
| Address-relation directed edges | 6,393,246 |
| Device-relation directed edges | 970,760 |
| Combined directed edges (deduplicated) | 7,266,146 |
| V2 address-history snapshots | 470,193 |
| V2 device-history snapshots | 85,593 |

`TransactionDT` is a **relative elapsed-time variable**, not a known real-world timestamp. We sort by `(TransactionDT, TransactionID)` to establish stable node positions, then assign chronological splits by time cutoffs (70% / 15% / 15% by row position, with equal-time transactions kept together). Fraud rates were about 3.52% / 3.43% / 3.48% respectively.

### Shared transaction features

We intentionally give the neural baselines and XGBoost the same underlying transaction information:

- **Numeric:** Train-only 0.1%–99.9% quantile clipping, median imputation, and z-score statistics, processed column-by-column to reduce peak RAM.
- **Categorical:** Train-only vocabularies; `0 = MISSING`, `1 = UNKNOWN`, and `2+ = observed in training`. Neural networks use per-column learned embeddings; XGBoost uses native categorical splits.
- **Engineered features:** Indicators such as `has_device_info`, `same_email_domain`, and whether both email domains are available.
- **Excluded:** `TransactionID`, the target `isFraud`, and raw `TransactionDT` as direct predictors. We do not insert target-derived full-dataset fraud rates into features.

We reuse the same masks, preprocessing, numeric/categorical data, and train-only class imbalance weighting across relevant model comparisons.

## Graph design

### V1: Directed homogeneous transaction graphs

Each **node is a transaction**. We infer an entity relationship if two transactions share either `(card1, addr1)` or `(card1, DeviceInfo)`. These are **proxies**, not verified customer/device identities.

For destination transaction `v`, we connect at most **K = 20** earlier transactions from the same inferred entity within **W = 30 days**:

```text
Historical transaction u ──────> Current transaction v
           only if: TransactionDT[u] < TransactionDT[v]
                    same composite key
                    within 30 days
                    among 20 most recent predecessors
```

Equal-time transactions are not connected. The union graph deduplicates edges and can contain up to 40 incoming edges per node (20 per relation). The one-layer GraphSAGE model uses mean aggregation and sampled incoming neighbors; supervised loss is computed **only on seed nodes**, never on sampled context-node labels.

### V2: Typed temporal history snapshots

Instead of mixing address and device links into one edge type, **TemporalHeteroFraud** creates a *transaction-specific snapshot of strictly historical activity* for each available relation:

```text
Past transaction(s) ──> address_history(v) ──> transaction v
Past transaction(s) ──> device_history(v)  ──> transaction v
```

There are **three node types** (`transaction`, `address_history`, `device_history`) and **four directed edge types**. A two-stage relation-aware network:

1. Uses the shared numeric + categorical `NodeEncoder` to encode historical transactions.
2. Aggregates address and device predecessors **separately**, with separate learned `SAGEConv` weights and historical-count embeddings.
3. Sends each typed context to its owning target transaction, concatenates target + address + device signals, and classifies fraud.

**Important distinction:** V2 is a *causal heterogeneous history-snapshot graph*, **not** a conventional persistent card/device/entity graph. This deliberate construction avoids static bidirectional connections that could transmit future transaction features into earlier predictions. The historical feature availability at event time should still be audited for a deployment setting.

The design and commands are documented in [`docs/heterogeneous_v2.md`](docs/heterogeneous_v2.md).

## Repository layout

```text
relational-fraud-detection/
├── README.md
├── pyproject.toml
├── uv.lock
├── src/fraud_graph/
│   ├── config.py                 # Experiment configuration
│   ├── paths.py                  # Canonical data/artifact locations
│   ├── preprocessing.py          # Train-only feature processing
│   ├── graph.py                  # V1 temporal edges and validation
│   ├── hetero_v2.py              # Typed temporal snapshot topology
│   ├── models.py                 # Shared NodeEncoder, MLP, GraphSAGE
│   ├── hetero_models.py          # TemporalHeteroFraud
│   ├── data.py                   # Memory-mapped arrays and loaders
│   ├── training.py               # Optimization and early stopping
│   └── utils.py
├── scripts/
│   ├── download_data.py
│   ├── build_dataset.py
│   ├── build_graphs.py
│   ├── build_hetero_v2.py
│   ├── train_mlp.py
│   ├── train_gnn.py
│   ├── train_hetero_v2.py
│   ├── train_xgboost.py
│   ├── migrate_legacy_artifacts.py
│   └── compare_results.py
├── notebooks/                   # EDA, graph design, baselines, GNNs
├── tests/                       # Graph, preprocessing, model, V2 tests
├── docs/                        # Methodology and V2 design
├── data/{raw,interim,processed}/
└── artifacts/{models,metrics,figures}/
```

**Data and checkpoints are intentionally excluded from Git.** Raw Kaggle CSVs, generated feature matrices, `.pt` graphs, and model weights are locally generated or migrated, not shipped with this repository.

## Reproduction

### 1. Set up the environment

Python 3.11 + [`uv`](https://docs.astral.sh/uv/) recommended:

```bash
uv sync --group dev
uv run pytest -q
uv run python -c "import torch, torch_geometric, xgboost; print(torch.__version__, torch_geometric.__version__, xgboost.__version__)"
```

For PyG's `NeighborLoader`, you may need a matching **`pyg-lib`** (or `torch-sparse`) wheel for the *active* PyTorch/CUDA build. See the [official PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html). The Python environment used by Lightning notebooks and the terminal must match.

### 2. Prepare datasets

Accept the [Kaggle competition rules](https://www.kaggle.com/competitions/ieee-fraud-detection), authenticate Kaggle, and run:

```bash
uv run python scripts/download_data.py
uv run python scripts/build_dataset.py
```

Alternatively, place `train_transaction.csv` and `train_identity.csv` under `data/raw/`. Building the full feature matrix is memory intensive; the pipeline processes numeric columns individually and saves memory-mapped arrays.

If you're migrating **trusted** artifacts from our earlier Lightning notebook project, you can avoid rebuilding:

```bash
uv run python scripts/migrate_legacy_artifacts.py \
  --source ../gnn-architecture-lab/notebooks/data/processed
```

By default this creates **symlinks** to existing large arrays. Keep the original project folder; use `--copy` if you need an independent copy. The migration can recover `node_transaction_ids.npy` and `transaction_times.npy` when the raw transaction CSV is available under `data/raw/`.

### 3. Run V1 experiments

**On a fresh build**, construct graphs:

```bash
uv run python scripts/build_graphs.py --relation all --window-days 30 --k 20
```

**If you migrated graph files as symlinks, do not rerun that builder over them.** It could overwrite files in the original project.

Train baselines:

```bash
uv run python scripts/train_mlp.py --seed 42
uv run python scripts/train_gnn.py --graph address --seed 42
uv run python scripts/train_gnn.py --graph device --seed 42
uv run python scripts/train_gnn.py --graph combined --seed 42
uv run python scripts/train_xgboost.py --rounds 5000 --patience 200 --seed 42
```

These scripts select neural epochs using validation AP, while XGBoost uses validation AUCPR for tree-round early stopping. Final reported AP is computed with `sklearn.average_precision_score`.

### 4. Build and train V2

```bash
uv run python scripts/build_hetero_v2.py
uv run python scripts/train_hetero_v2.py --smoke-test --batch-size 128
uv run python scripts/train_hetero_v2.py \
  --batch-size 512 --max-epochs 15 --patience 3 --seed 42
```

The smoke test verifies sampled two-hop history completeness and finite forward/backward results before full training. V2 stores its topology in `data/processed/hetero_structure_v2.pt` and model/metric outputs in `artifacts/`.

### 5. Inspect locally saved run metrics

```bash
uv run python scripts/compare_results.py
cat artifacts/metrics/hetero_v2_seed42.json
```

`compare_results.py` reports JSON metrics **actually generated on your machine**; it does not fabricate the original notebook scores in the results table above.

## Interpretation and limitations

**What happened?** Repeated experimentation showed that XGBoost captured fraud-discriminative information more effectively than the GNN configurations tested. A likely contributing factor is that IEEE-CIS already provides rich engineered tabular variables and our graph links are imperfect identity proxies. Additionally, mean-based aggregation may discard some activity-count information. These are **hypotheses**, not isolated causal explanations.

**What we cannot claim:**

- That GNNs are inherently inferior to XGBoost on this dataset or in general.
- That adding a hetero graph should necessarily improve performance.
- That our ROC-AUC values are directly comparable to the official Kaggle leaderboard: our split is a separate chronological holdout of the released *training* labels.
- That the final test period was a pristine confirmatory holdout throughout development: its scores and full-dataset EDA were consulted during later exploratory model design.

**Rigor improvements needed for a research claim:** Repeat neural runs across multiple seeds, audit point-in-time availability of masked features, freeze a fresh forward-time holdout, and evaluate operational measures (e.g., precision/recall at a review-capacity budget) alongside AP/ROC-AUC.

## Next experiment (proposed, not implemented)

Test whether **explicit causal historical aggregates** add value to a strong tabular model:

- Prior same-entity transaction counts over 1, 7, and 30 days.
- Recency since prior transaction, recent amount statistics, and distinct recent device/address counts.
- Retrain **XGBoost + point-in-time relational features**, then compare with the frozen XGBoost control on validation; use a fresh later holdout for confirmatory evidence.

A positive result would show that historical relations contain signal but our message passing wasn't the best way to exploit it. A null result would suggest those particular relationships are largely redundant with available tabular features. Neither outcome should be prejudged.

## References

- [IEEE-CIS Fraud Detection — Kaggle dataset/competition](https://www.kaggle.com/competitions/ieee-fraud-detection)
- [PyTorch Geometric documentation](https://pytorch-geometric.readthedocs.io/)
- [Stanford CS224W — Machine Learning with Graphs](https://snap.stanford.edu/class/cs224w-2025/)
- [BRIGHT: Graph Neural Networks in Real-Time Fraud Detection](https://arxiv.org/abs/2205.13084) — related causal/real-time graph-design work; **not** a like-for-like comparison of our split.
- [Why Do Tree-Based Models Still Outperform Deep Learning on Typical Tabular Data?](https://arxiv.org/abs/2207.08815) — background on tabular inductive biases, not an explanation established by our experiments.

---

**Bottom line:** This repository documents an end-to-end, leakage-conscious graph ML workflow *and* a negative experimental result. The engineering accomplishment is transforming raw mixed-type data into reproducible temporal and heterogeneous graph inputs; the modeling lesson is that **the information encoded by the graph matters more than architectural sophistication alone**.
