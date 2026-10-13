"""Build and persist address, device, and combined causal graphs."""

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fraud_graph.config import GraphConfig
from fraud_graph.graph import (
    build_temporal_edges, chronological_frame, union_edges, validate_temporal_edges,
)
from fraud_graph.paths import PROCESSED_DIR, RAW_DIR
from fraud_graph.utils import save_json

RELATION_FILES = {
    "address": "edge_index_v1.pt",
    "device": "edge_index_device_v1.pt",
    "combined": "edge_index_combined_v1.pt",
}


def main(raw_dir: Path, processed_dir: Path, relation: str, window_days: int, k: int):
    cfg = GraphConfig(window_days=window_days, max_neighbors=k)
    processed_dir.mkdir(parents=True, exist_ok=True)
    ids = np.load(processed_dir / "node_transaction_ids.npy")
    times = np.load(processed_dir / "transaction_times.npy")
    needs_address = relation in {"address", "combined", "all"}
    needs_device = relation in {"device", "combined", "all"}
    cols = ["TransactionID", "TransactionDT", "card1"]
    if needs_address:
        cols.append("addr1")
    frame = pd.read_csv(raw_dir / "train_transaction.csv", usecols=cols)
    if needs_device:
        identity = pd.read_csv(raw_dir / "train_identity.csv",
                               usecols=["TransactionID", "DeviceInfo"])
        if not identity.TransactionID.is_unique:
            raise ValueError("Identity TransactionIDs must be unique")
        frame = frame.merge(identity, on="TransactionID", how="left", validate="one_to_one")
    frame = chronological_frame(frame)
    if not np.array_equal(frame.TransactionID.to_numpy(), ids):
        raise ValueError("Graph node order disagrees with processed feature row order")
    if not np.array_equal(frame.TransactionDT.to_numpy(), times):
        raise ValueError("Graph times disagree with processed feature times")

    summaries = {}
    if needs_address:
        print("Building card1 + addr1 relation...", flush=True)
        e_addr = build_temporal_edges(frame, ("card1", "addr1"), cfg)
        summaries["address"] = validate_temporal_edges(
            e_addr, frame, cfg, max_in_degree=k)
        torch.save(e_addr, processed_dir / RELATION_FILES["address"])
        print("Address:", summaries["address"])
    if needs_device:
        print("Building card1 + DeviceInfo relation...", flush=True)
        e_device = build_temporal_edges(frame, ("card1", "DeviceInfo"), cfg)
        summaries["device"] = validate_temporal_edges(
            e_device, frame, cfg, max_in_degree=k)
        torch.save(e_device, processed_dir / RELATION_FILES["device"])
        print("Device:", summaries["device"])
    if relation in {"combined", "all"}:
        print("Unioning and deduplicating...", flush=True)
        e_union = union_edges(e_addr, e_device, num_nodes=len(frame))
        summaries["combined"] = validate_temporal_edges(
            e_union, frame, cfg, max_in_degree=2 * k)
        torch.save(e_union, processed_dir / RELATION_FILES["combined"])
        print("Combined:", summaries["combined"])
    save_json(processed_dir / "graph_metadata.json", {
        "window_days": window_days,
        "max_neighbors_per_relation": k,
        "relations_built": summaries,
    })
    del frame
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--relation", choices=["address", "device", "combined", "all"],
                        default="all")
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--k", type=int, default=20)
    args = parser.parse_args()
    main(args.raw_dir, args.processed_dir, args.relation, args.window_days, args.k)
