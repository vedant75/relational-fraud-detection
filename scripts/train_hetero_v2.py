"""Run V2 two-hop, three-node-type temporal heterogeneous GNN.

This script only reads frozen V1 matrices/edges, and writes NEW V2 artifacts.
One smoke-test batch can be run first with --smoke-test; it never selects on test.
"""

import argparse
import copy
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch_geometric.loader import NeighborLoader

from fraud_graph.data import load_processed
from fraud_graph.hetero_models import TemporalHeteroFraud
from fraud_graph.hetero_v2 import (
    TX, EDGE_TYPES, load_structure,
    ADDR_IN, ADDR_OUT, ADDR_HISTORY, DEVICE_IN, DEVICE_OUT, DEVICE_HISTORY,
)
from fraud_graph.models import NodeEncoder, embedding_dimensions
from fraud_graph.paths import MODEL_DIR, METRICS_DIR, PROCESSED_DIR, ensure_output_dirs
from fraud_graph.training import positive_weight
from fraud_graph.utils import save_json, set_seed


def attach_node_data(graph, arrays):
    """Avoid serializing the gigabyte-scale feature matrix inside graph topology."""
    if graph[TX].num_nodes != len(arrays["y"]):
        raise ValueError("Hetero node count differs from frozen feature matrices")
    graph[TX].x_num = torch.from_numpy(arrays["numeric"])
    graph[TX].x_cat = torch.from_numpy(arrays["categorical"])
    graph[TX].y = torch.from_numpy(arrays["y"]).float()
    for split in ("train", "val", "test"):
        setattr(graph[TX], f"{split}_mask", torch.from_numpy(arrays[f"{split}_mask"]))
    return graph


def make_loaders(graph, batch_size, seed):
    # Directional two-hop sampler traverses against edge direction from each
    # transaction seed, first context -> transaction, then historical transaction
    # -> context. No full graph is moved to GPU.
    sizes = {edge_type: [-1, -1] for edge_type in EDGE_TYPES}
    loaders = {}
    for name in ("train", "val", "test"):
        loaders[name] = NeighborLoader(
            graph,
            num_neighbors=sizes,
            input_nodes=(TX, graph[TX][f"{name}_mask"]),
            batch_size=batch_size,
            shuffle=name == "train",
            num_workers=0,
            subgraph_type="directional",
            generator=torch.Generator().manual_seed(seed) if name == "train" else None,
        )
    return loaders


def make_model(metadata, device, hidden_dim=128, encoder_dropout=0.1, dropout=0.2):
    dims = embedding_dimensions(metadata["category_sizes"])
    encoder = NodeEncoder(
        len(metadata["numeric_cols"]), metadata["categorical_cols"],
        metadata["category_sizes"], dims, hidden_dim=hidden_dim,
        dropout=encoder_dropout,
    )
    return TemporalHeteroFraud(encoder, hidden_dim=hidden_dim, dropout=dropout).to(device), dims


def forward_seed(model, batch, device):
    batch = batch.to(device)
    logits = model(batch)
    # PyG orders each type's seed nodes before its sampled context nodes.
    seeds = batch[TX].batch_size
    assert logits.numel() == batch[TX].num_nodes
    return logits[:seeds], batch[TX].y[:seeds]


def one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    loss_sum = 0.0
    n_sum = 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        logits, labels = forward_seed(model, batch, device)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        count = labels.size(0)
        n_sum += count
        loss_sum += loss.item() * count
    return loss_sum / n_sum


@torch.no_grad()
def score(model, loader, criterion, device):
    model.eval()
    labels_all, probs_all = [], []
    loss_sum, n_sum = 0.0, 0
    for batch in loader:
        logits, labels = forward_seed(model, batch, device)
        loss = criterion(logits, labels)
        count = labels.size(0)
        n_sum += count
        loss_sum += loss.item() * count
        labels_all.append(labels.detach().cpu().numpy())
        probs_all.append(torch.sigmoid(logits).detach().cpu().numpy())
    labels = np.concatenate(labels_all)
    probs = np.concatenate(probs_all)
    return {
        "loss": loss_sum / n_sum,
        "pr_auc": float(average_precision_score(labels, probs)),
        "roc_auc": float(roc_auc_score(labels, probs)),
    }


def main(args):
    ensure_output_dirs()
    set_seed(args.seed)
    arrays, metadata = load_processed(args.processed_dir)
    topology_path = args.topology or args.processed_dir / "hetero_structure_v2.pt"
    graph = attach_node_data(load_structure(topology_path), arrays)
    print("Node types:", graph.node_types)
    print("Edge types:", graph.edge_types)
    print("Nodes:", {t: graph[t].num_nodes for t in graph.node_types}, flush=True)

    loaders = make_loaders(graph, args.batch_size, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)
    model, dims = make_model(metadata, device, hidden_dim=args.hidden_dim)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            positive_weight(arrays["y"], arrays["train_mask"]),
            dtype=torch.float32, device=device,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    if args.smoke_test:
        batch = next(iter(loaders["train"]))
        print("Dry-run seed nodes:", batch[TX].batch_size)
        print("Dry-run node counts:", {t: batch[t].num_nodes for t in batch.node_types})
        print("Dry-run edge counts:", {str(et): batch[et].edge_index.shape[1] for et in batch.edge_types})
        # Verify that ALL predecessors were recovered by the two-hop loader
        # for the history snapshots belonging to this batch of seeds.
        for node_type, et_in, et_out in [
            (ADDR_HISTORY, ADDR_IN, ADDR_OUT),
            (DEVICE_HISTORY, DEVICE_IN, DEVICE_OUT),
        ]:
            to_seed = batch[et_out].edge_index
            if to_seed.numel() == 0:
                continue
            assert bool(torch.all(to_seed[1] < batch[TX].batch_size))
            relevant_context = to_seed[0]
            recovered = torch.bincount(
                batch[et_in].edge_index[1], minlength=batch[node_type].num_nodes
            )
            original_count = torch.expm1(
                batch[node_type].log_count.squeeze(-1) * math.log1p(graph.graph_k)
            ).round().long()
            assert torch.equal(recovered[relevant_context], original_count[relevant_context]), (
                f"NeighborLoader did not return full 2-hop history for {node_type}"
            )
        print("Two-hop history completeness: PASSED")
        logits, labels = forward_seed(model, batch, device)
        loss = criterion(logits, labels)
        loss.backward()
        print("Seed logits:", tuple(logits.shape))
        print("Loss:", float(loss.item()))
        print("Address grad norm:", float(model.addr_historical_mean.lin_l.weight.grad.norm().item()))
        dev_grad = model.device_historical_mean.lin_l.weight.grad
        print("Device grad norm:", None if dev_grad is None else float(dev_grad.norm().item()))
        print("Finite:", bool(torch.isfinite(logits).all()))
        return

    best_score = -float("inf")
    best_state = None
    best_epoch = 0
    no_improvement = 0
    history = []
    for epoch in range(1, args.max_epochs + 1):
        start = time.perf_counter()
        train_loss = one_epoch(model, loaders["train"], optimizer, criterion, device)
        val = score(model, loaders["val"], criterion, device)
        record = {
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val["loss"], "val_pr_auc": val["pr_auc"],
            "val_roc_auc": val["roc_auc"],
            "seconds": time.perf_counter() - start,
        }
        history.append(record)
        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
            f"val_loss={val['loss']:.4f} | val_PR_AUC={val['pr_auc']:.4f} | "
            f"val_ROC_AUC={val['roc_auc']:.4f} | time={record['seconds']:.1f}s",
            flush=True,
        )
        if val["pr_auc"] > best_score:
            best_score = val["pr_auc"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= args.patience:
                print(f"Early stopping. Best epoch: {best_epoch}", flush=True)
                break
    assert best_state is not None
    model.load_state_dict(best_state)

    # Test is evaluated only after validation has selected a fixed checkpoint.
    test_metrics = score(model, loaders["test"], criterion, device)
    results = {
        "model": "TemporalHeteroFraud",
        "graph": "typed_address_and_device_history_snapshots",
        "seed": args.seed,
        "best_epoch": best_epoch,
        "val_pr_auc": best_score,
        "test_pr_auc": test_metrics["pr_auc"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_loss": test_metrics["loss"],
        "hidden_dim": args.hidden_dim,
        "batch_size": args.batch_size,
        "num_address_snapshots": graph["address_history"].num_nodes,
        "num_device_snapshots": graph["device_history"].num_nodes,
    }
    label = f"hetero_v2_seed{args.seed}"
    torch.save(
        {"model_state_dict": model.state_dict(), "results": results, "embedding_dims": dims},
        MODEL_DIR / f"{label}.pt",
    )
    pd.DataFrame(history).to_csv(METRICS_DIR / f"{label}_history.csv", index=False)
    save_json(METRICS_DIR / f"{label}.json", results)
    print("FINAL RESULTS:", results, flush=True)
    print("Existing XGBoost reference: test PR-AUC 0.5407, ROC-AUC 0.8918", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--topology", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--smoke-test", action="store_true")
    main(parser.parse_args())
