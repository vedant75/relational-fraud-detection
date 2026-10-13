"""Build the canonical node feature matrices and temporal split.

Expected IEEE-CIS rows (original user's run): 590540, 397 numerical, 37
categorical after adding 3 engineered indicators. Resource intensive; run once.
"""

import argparse
import gc
from pathlib import Path

import pandas as pd

from fraud_graph.paths import RAW_DIR, PROCESSED_DIR
from fraud_graph.preprocessing import build_processed_data


def main(raw_dir: Path, processed_dir: Path):
    transaction_file = raw_dir / "train_transaction.csv"
    identity_file = raw_dir / "train_identity.csv"
    if not transaction_file.exists() or not identity_file.exists():
        raise FileNotFoundError("Run scripts/download_data.py first")
    print("Loading raw transaction table...", flush=True)
    transactions = pd.read_csv(transaction_file, low_memory=False)
    print("Loading identity table...", flush=True)
    identity = pd.read_csv(identity_file, low_memory=False)
    if not transactions.TransactionID.is_unique or not identity.TransactionID.is_unique:
        raise ValueError("TransactionID must be unique in both input tables")
    model_df = transactions.merge(identity, on="TransactionID", how="left",
                                  validate="one_to_one")
    assert len(model_df) == len(transactions)
    del transactions, identity
    gc.collect()
    print("Building numerical and categorical matrices, one column at a time...", flush=True)
    info = build_processed_data(model_df, processed_dir)
    del model_df
    gc.collect()
    print("Built:")
    print("Numeric:", info["numeric_shape"], "Categorical:", info["categorical_shape"])
    print("Splits:", info["split_sizes"])
    print("Fraud rates:", info["fraud_rate_by_split"])
    print("Cutoffs:", info["cutoffs"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    args = parser.parse_args()
    main(args.raw_dir, args.processed_dir)
