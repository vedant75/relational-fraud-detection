# V2: Causal temporal heterogeneous fraud detection

## Research question

Can preserving separately learned card/address and card/device historical context improve
fraud classification beyond the frozen V1 one-layer projected GraphSAGE models and
XGBoost? **Do not assume the answer will be yes.**

### Frozen observations (not predictions about V2)

| Model | Validation AP / PR-AUC | Test AP / PR-AUC | Test ROC-AUC |
| --- | ---: | ---: | ---: |
| Homogeneous GraphSAGE: address | 0.4838 | 0.4538 | 0.8599 |
| Homogeneous GraphSAGE: device | 0.4994 | 0.4657 | 0.8360 |
| Homogeneous GraphSAGE: union | 0.4939 | 0.4670 | 0.8612 |
| XGBoost, no graph | 0.6723 | 0.5407 | 0.8918 |

### Why not a static transaction/card bipartite graph?

A naively bidirectional static graph can propagate a feature from a *future*
transaction through a shared card node into an *earlier* transaction. That
invalidates a realistic forward-time evaluation even if labels are excluded.

Instead, V2 creates a **history snapshot for each transaction/relation** only
when there are earlier qualifying transactions. This is a *temporal heterogeneous
snapshot graph*, **not** a conventional persistent entity graph:

```
historical transaction u
         |    (TransactionDT[u] < TransactionDT[v])
         v
address_history_snapshot(v) --address_context_for--> transaction v

historical transaction w
         |
         v
device_history_snapshot(v) --device_context_for----> transaction v
```

There are 3 node types (`transaction`, `address_history`, `device_history`)
and 4 typed directed edge types. Every context has exactly one owning target
transaction. Construction reuses the V1 30-day/K=20 predecessor sets; neither
new raw CSV parsing nor retraining V1 models is required.

Two-stage relation-aware architecture:

1. **Stage 1:** Encode raw historical transactions using the exact same numeric
   + learned-categorical-embedding `NodeEncoder` used in V1. Aggregate historical
   features separately by address/device history snapshot with `SAGEConv` mean,
   add a learnable representation of historical count (`log1p(count)/log1p(20)`).
2. **Stage 2:** Push each snapshot to its single target transaction through a
   relation-specific `SAGEConv`. Concatenate root transaction embedding and
   both relation signals, then classify via MLP. Missing history gets zero
   message signal for that relation.

This is a typed heterogeneous GNN that can learn distinct relational weights.
It is **not** a GIN, TGN, R-GCN or standard globally shared card-entity model.
Mean aggregators can still lose information, partly mitigated by count input.

## Lightning AI: exact commands

All commands below start from the existing Lightning terminal. Ensure your
previously migrated files still resolve in `data/processed` and leave the
`gnn-architecture-lab` project intact (the processed matrices are symlinks).

1. Copy the three new source/modules, two new scripts, and new test/document
   files into their matching locations under `relational-fraud-detection/`.
2. In the terminal:

   ```bash
   cd /teamspace/studios/this_studio/relational-fraud-detection
   uv run pytest -q
   ```

3. Build **a NEW V2 graph file** (no existing V1 graph files overwritten):

   ```bash
   uv run python scripts/build_hetero_v2.py
   ```

   This reads `edge_index_v1.pt`, `edge_index_device_v1.pt`, and
   `transaction_times.npy`, validates historical ordering, and writes
   `data/processed/hetero_structure_v2.pt` with a sidecar stats JSON.

4. **Mandatory smoke test**, ideally before a long training run:

   ```bash
   uv run python scripts/train_hetero_v2.py --smoke-test --batch-size 128
   ```

   Expected: 3 node types, 4 edge types, seed logits of shape `(128,)`,
   finite loss, and a finite nonzero history-layer gradient norm. This
   command only tests a single minibatch and saves no checkpoint.

   If you get `NeighborSampler requires either 'pyg-lib' or 'torch-sparse'`,
   install a `pyg-lib` wheel matching the **actual new project's** PyTorch
   version, not the old project's interpreter. Check it with:

   ```bash
   uv run python -c "import torch; print(torch.__version__)"
   uv run python -c "import pyg_lib; print('pyg-lib available')"
   ```

   In this Lightning environment, PyTorch was reported as 2.14.1+cu130;
   prebuilt extension indexes often use the matching major/minor Torch and
   CUDA version, but check compatibility with your installed build before
   installing anything. Do not use the old `gnn-architecture-lab/.venv`.

5. Once the smoke test passes, train:

   ```bash
   uv run python scripts/train_hetero_v2.py \
     --batch-size 512 --max-epochs 15 --patience 3 --seed 42
   ```

   Training runs on CPU if CUDA is unavailable, like your V1 runs.
   It can take multiple minutes per epoch. Validation AP selects the best
   epoch. **Only after** selection, the script evaluates test once.

6. Review results and compare:

   ```bash
   uv run python scripts/compare_results.py
   cat artifacts/metrics/hetero_v2_seed42.json
   ```

   Saved artifacts (Git ignored by design):
   - `data/processed/hetero_structure_v2.pt`
   - `data/processed/hetero_structure_v2.json`
   - `artifacts/models/hetero_v2_seed42.pt`
   - `artifacts/metrics/hetero_v2_seed42.json`
   - `artifacts/metrics/hetero_v2_seed42_history.csv`

7. Commit **code/docs/tests only**, as the `.gitignore` already excludes
   datasets and checkpoints:

   ```bash
   git status --short
   git add src/fraud_graph/hetero_v2.py src/fraud_graph/hetero_models.py \
     scripts/build_hetero_v2.py scripts/train_hetero_v2.py \
     tests/test_hetero_v2.py docs/heterogeneous_v2.md
   git diff --cached --stat
   git commit -m "Add leakage-safe temporal heterogeneous fraud experiment"
   git push origin main
   ```

## How to interpret V2

Compare V2 test AP to 0.4670 (best homogeneous GNN) and 0.5407 (XGBoost).
Do not adjust V2 on the basis of test performance. The original test period
has already been consulted repeatedly across V1 design choices, so the V2
score is an exploratory comparison rather than a pristine confirmatory holdout.
A new later-time holdout, or a second temporal dataset, is needed for stronger
claims of generalization. Because these models differ in seed and training
schedule, later use repeated seeds to estimate uncertainty.

Do not use future fraud labels as features. The context nodes only pool
transaction features available at the event time; label-based history features
would require a separate point-in-time label-availability model.
