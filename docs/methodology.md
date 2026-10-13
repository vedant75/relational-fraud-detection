# Graph and evaluation methodology

## Dataset grain

One node is one IEEE-CIS `train_transaction.csv` transaction. The optional
`train_identity.csv` is left-joined on unique `TransactionID`. No row is dropped
for missing identity. DeviceInfo coverage is only about 20.1% in the original
train table. The target is binary `isFraud` (~3.5% positives).

## Relations tested

- **Address relation**: `(card1, addr1)` — higher coverage, inferred actor/account proxy.
- **Device relation**: `(card1, DeviceInfo)` — narrower, only transactions with identity data.
  `DeviceInfo` is a device/UA/model signature, NOT a guaranteed unique device ID.
- **Combined**: deduplicated union of both. Same directed edge may occur in both.

For each transaction, select up to **20** strictly earlier transactions with the
same composite key, at most **30 days** old. Both graph relations use identical
horizon/cap for a controlled comparison. Equal timestamps cannot imply ordering
and must never be linked to each other. The combined incoming degree can be 40.

Graphs are *directed*: old transaction → current transaction. PyG's one-layer
`NeighborLoader(num_neighbors=[-1])` gives exactly the incoming message history
for seed transactions. The first `batch.batch_size` node IDs in the sampled
batch are the targets; other nodes are feature-only context. The target labels
of context nodes are not supplied to the model.

## Chronology and preprocessing

Rows are sorted by `TransactionDT` and `TransactionID` before defining integer
node IDs. Time is a *relative offset*, not a wall-clock timestamp. A split
is made at the timestamp of the 70th/85th percentile by row position:

- Train 413,378 transactions
- Validation 88,581
- Test 88,581

The exact row counts above are from the original run. Each cutoff is exclusive
on the preceding split, so equal-time transactions always stay together.

Numeric input features are train-only quantile clipped at 0.1/99.9%, train-only
median-imputed, then train-only z-scored (population SD). All 397 numeric features
were processed one column at a time to prevent RAM crashes. Categorical fields
are train-only vocabulary-coded: 0 = missing, 1 = unseen category, 2+ = known.
37 categorical fields are fed through separate learned embedding tables.

Feature construction never uses `isFraud` group rates, future counts, or future
category vocabularies. `TransactionID`, `TransactionDT` and target are excluded
from input matrices.

## Models and metrics

All neural runs have a shared 397-numeric/37-categorical `NodeEncoder`, 128-
dimensional output, similar classification head, weighted BCE (positive weight
computed on training labels only), AdamW at 1e-3, batch size 1024, up to 15
epochs, patience 3, and validation sklearn `average_precision_score` selection.

- **MLP**: ignores graph altogether.
- **GraphSAGE**: 1 mean-aggregation layer; all immediate incoming predecessors.
- **XGBoost**: native categories, histogram tree, depth 8, learning rate .05,
  weight ratio, up to 5000 rounds with validation XGBoost AUCPR early stopping
  patience 200. Selected best iteration 3596 in the original run.

**Primary metric** = sklearn Average Precision (reported as PR-AUC).
**Secondary** = ROC-AUC. XGBoost internal `aucpr` may differ slightly from
sklearn AP; early stopping uses XGB aucpr, final model comparison uses sklearn AP.

## Experiment validity qualifications

This is an **exploratory benchmark**, not a sealed external test. In the
original interactive work, we inspected full-dataset fraud EDA and test results
from earlier models before selecting subsequent graph hypotheses. We therefore
must not claim the final test set remained completely untouched throughout all
research decisions. For a publication-grade estimate, freeze all hypotheses,
then test on a newly held-out later time period or independent data. Also, the
MLP/first address GraphSAGE runs used uncontrolled random seeds and varied
substantially between reruns; repeat each neural experiment with at least
three seeds and report mean/standard deviation if seeking statistical claims.

A two-layer causal graph with old→new directed edges is also conceptually safe
for feature flow, but this project deliberately tested one layer only. In
prospective deployments, verify availability of each feature at transaction
time. Some masked IEEE-CIS fields may encode delayed/history-derived data whose
true time availability is not documented; point-in-time feature audit remains
important.

## Interpretation

Device relation beat address relation by ~0.012 test AP, but union added only
~0.0013 AP while using ~7.5x as many edges as device-only. XGBoost still
outperformed the best GraphSAGE on the test set (0.5407 vs 0.4670). This does
not rule out all graph models: it is evidence about these inferred relations
and this single-layer homogeneous mean-GraphSAGE setup.
