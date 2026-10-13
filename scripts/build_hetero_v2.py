"""Build causal heterogeneous history-snapshot topology from frozen V1 graphs.

Does not touch/overwrite the original .pt symlinks or CSV files.
"""

import argparse
import gc
from pathlib import Path

import numpy as np
import torch

from fraud_graph.hetero_v2 import make_hetero_structure, validate_hetero_structure
from fraud_graph.paths import PROCESSED_DIR
from fraud_graph.utils import save_json


def main(processed_dir: Path, output: Path, k: int, window_days: int):
    required = [
        "edge_index_v1.pt", "edge_index_device_v1.pt", "transaction_times.npy",
    ]
    missing = [name for name in required if not (processed_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing {missing}; migrate legacy data first")
    times = np.load(processed_dir / "transaction_times.npy", mmap_mode="r")
    addr = torch.load(processed_dir / "edge_index_v1.pt", map_location="cpu", weights_only=True)
    dev = torch.load(processed_dir / "edge_index_device_v1.pt", map_location="cpu", weights_only=True)
    print(f"Read {addr.shape[1]:,} address edges, {dev.shape[1]:,} device edges", flush=True)
    graph = make_hetero_structure(addr, dev, times, k=k, window_days=window_days)
    del addr, dev
    gc.collect()
    stats = validate_hetero_structure(graph, times, k=k, window_days=window_days)
    if output.is_symlink():
        raise ValueError("Refusing to overwrite a symlink; specify a new output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph, output)
    save_json(output.with_suffix(".json"), {**stats, "window_days": window_days, "k": k})
    print("Heterogeneous topology saved:", output)
    print("Stats:", stats)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--window-days", type=int, default=30)
    args = p.parse_args()
    output = args.output or (args.processed_dir / "hetero_structure_v2.pt")
    main(args.processed_dir, output, args.k, args.window_days)
