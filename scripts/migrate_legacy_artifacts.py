"""Convert earlier notebook artifacts into this repo's canonical file layout.

Requires trusted, locally-created preprocessing.pkl; never unpickle unknown files.
Does not regenerate the large feature matrices or graph edges.
"""

import argparse
import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_graph.paths import PROCESSED_DIR, RAW_DIR
from fraud_graph.utils import save_json

FILES = ["numeric_features.npy", "categorical_features.npy", "y.npy",
         "train_mask.npy", "val_mask.npy", "test_mask.npy",
         "edge_index_v1.pt", "edge_index_device_v1.pt",
         "edge_index_combined_v1.pt"]


def main(source: Path, target: Path, raw: Path, copy: bool):
    source = source.resolve()
    target.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        src = source / name
        dst = target / name
        if src.exists() and src != dst and not dst.exists():
            if copy:
                shutil.copy2(src, dst)
            else:
                dst.symlink_to(src)
            print("Imported", name)
    metadata = target / "preprocessing.json"
    if not metadata.exists():
        legacy_file = source / "preprocessing.pkl"
        if not legacy_file.exists():
            raise FileNotFoundError("Missing preprocessing.pkl (or preprocessing.json)")
        with legacy_file.open("rb") as f:
            old = pickle.load(f)  # trusted artifacts ONLY
        numeric_stats_path = source / "numeric_stats.parquet"
        if numeric_stats_path.exists():
            stats_frame = pd.read_parquet(numeric_stats_path).set_index("feature")
            stats = stats_frame.to_dict(orient="index")
        else:
            stats = {}
        y = np.load(target / "y.npy")
        masks = {s: np.load(target / f"{s}_mask.npy") for s in ("train", "val", "test")}
        payload = {
            "numeric_cols": old["numeric_cols"],
            "categorical_cols": old["categorical_cols"],
            "category_sizes": old["category_sizes"],
            "category_maps": old["category_maps"],
            "numeric_stats": stats,
            "split_sizes": {s: int(m.sum()) for s, m in masks.items()},
            "fraud_rate_by_split": {s: float(y[m].mean()) for s, m in masks.items()},
            "numeric_shape": list(np.load(target / "numeric_features.npy", mmap_mode="r").shape),
            "categorical_shape": list(np.load(target / "categorical_features.npy", mmap_mode="r").shape),
            "cutoffs": {},
        }
        save_json(metadata, payload)
        print("Converted preprocessing.pkl → preprocessing.json")

    node_ids = target / "node_transaction_ids.npy"
    times = target / "transaction_times.npy"
    if not node_ids.exists() or not times.exists():
        raw_file = raw / "train_transaction.csv"
        if not raw_file.exists():
            print("Cannot recover node order without raw train_transaction.csv; "
                  "training works, but graph rebuilding requires these arrays")
        else:
            df = pd.read_csv(raw_file, usecols=["TransactionID", "TransactionDT"])
            df = df.sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)
            np.save(node_ids, df.TransactionID.to_numpy(dtype=np.int64))
            np.save(times, df.TransactionDT.to_numpy(dtype=np.int64))
            print("Recovered transaction node order and times")
    print("Migration finished. Canonical processed folder:", target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--copy", action="store_true",
                        help="Copy large artifacts; default is symlinks to save disk space")
    args = parser.parse_args()
    main(args.source, args.target, args.raw_dir, args.copy)
