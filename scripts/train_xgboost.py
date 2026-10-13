"""Strong non-graph baseline with native categorical XGBoost splits.

Recreates the user's XGBoost V0.5 settings: histogram trees, weight ratio,
5000 cap, validation AUCPR early stopping patience 200.
"""

import argparse
import gc
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

from fraud_graph.data import load_processed
from fraud_graph.paths import METRICS_DIR, MODEL_DIR, PROCESSED_DIR, ensure_output_dirs
from fraud_graph.training import positive_weight
from fraud_graph.utils import save_json


def build_matrix(arrays, idx, feature_names, feature_types, *, reference=None, labels=True):
    data = np.concatenate([
        np.asarray(arrays["numeric"][idx], dtype=np.float32),
        np.asarray(arrays["categorical"][idx], dtype=np.float32),
    ], axis=1)
    y = arrays["y"][idx] if labels else None
    dmatrix = xgb.QuantileDMatrix(
        data, label=y, feature_names=feature_names,
        feature_types=feature_types, enable_categorical=True, ref=reference,
    )
    del data
    gc.collect()
    return dmatrix


def main(processed_dir: Path, rounds: int, patience: int, seed: int):
    ensure_output_dirs()
    arrays, meta = load_processed(processed_dir)
    feature_names = meta["numeric_cols"] + meta["categorical_cols"]
    feature_types = ["q"] * len(meta["numeric_cols"]) + ["c"] * len(meta["categorical_cols"])
    train_idx = np.flatnonzero(arrays["train_mask"])
    val_idx = np.flatnonzero(arrays["val_mask"])
    test_idx = np.flatnonzero(arrays["test_mask"])

    print("Preparing train/validation histogram matrices...", flush=True)
    dtrain = build_matrix(arrays, train_idx, feature_names, feature_types)
    dval = build_matrix(arrays, val_idx, feature_names, feature_types, reference=dtrain)
    params = {
        "objective": "binary:logistic", "tree_method": "hist",
        "learning_rate": 0.05, "max_depth": 8, "subsample": 0.8,
        "colsample_bytree": 0.8, "min_child_weight": 5,
        "reg_lambda": 1.0, "reg_alpha": 0.0,
        "scale_pos_weight": positive_weight(arrays["y"], arrays["train_mask"]),
        "eval_metric": ["auc", "aucpr"], "seed": seed,
    }
    stop = xgb.callback.EarlyStopping(
        rounds=patience, metric_name="aucpr", data_name="validation",
        maximize=True, save_best=True,
    )
    eval_history = {}
    model = xgb.train(
        params, dtrain, num_boost_round=rounds,
        evals=[(dtrain, "train"), (dval, "validation")],
        callbacks=[stop], evals_result=eval_history, verbose_eval=200,
    )
    val_scores = model.predict(dval)
    val_ap = float(average_precision_score(arrays["y"][val_idx], val_scores))
    val_roc = float(roc_auc_score(arrays["y"][val_idx], val_scores))
    print(f"Validation AP={val_ap:.4f} ROC-AUC={val_roc:.4f}, "
          f"best iteration={model.best_iteration}")

    # Final evaluation only: do not use test results for model selection.
    print("Preparing test matrix...", flush=True)
    dtest = build_matrix(arrays, test_idx, feature_names, feature_types, reference=dtrain)
    test_scores = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))
    results = {
        "model": "XGBoost", "graph": None, "seed": seed,
        "best_iteration": model.best_iteration,
        "best_validation_xgb_aucpr": float(model.best_score),
        "val_pr_auc": val_ap, "val_roc_auc": val_roc,
        "test_pr_auc": float(average_precision_score(arrays["y"][test_idx], test_scores)),
        "test_roc_auc": float(roc_auc_score(arrays["y"][test_idx], test_scores)),
    }
    model.save_model(str(MODEL_DIR / f"xgboost_seed{seed}.json"))
    save_json(METRICS_DIR / f"xgboost_seed{seed}.json", results)
    save_json(METRICS_DIR / f"xgboost_seed{seed}_history.json", eval_history)
    print("Final XGBoost:", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--rounds", type=int, default=5000)
    parser.add_argument("--patience", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.processed_dir, args.rounds, args.patience, args.seed)
