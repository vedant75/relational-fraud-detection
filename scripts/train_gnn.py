"""Single-layer mean GraphSAGE; select the existing causal edge relation."""

import argparse
from pathlib import Path

import pandas as pd
import torch

from fraud_graph.config import TrainConfig
from fraud_graph.data import load_processed, make_pyg_loaders
from fraud_graph.models import NodeEncoder, GraphSAGEFraud, embedding_dimensions
from fraud_graph.paths import MODEL_DIR, METRICS_DIR, PROCESSED_DIR, ensure_output_dirs
from fraud_graph.training import create_optimizer_and_loss, evaluate, fit_early_stopping
from fraud_graph.utils import save_json, set_seed

GRAPH_NAMES = {
    "address": "edge_index_v1.pt",
    "device": "edge_index_device_v1.pt",
    "combined": "edge_index_combined_v1.pt",
}


def main(processed_dir: Path, graph_name: str, seed: int,
         epochs: int, patience: int, batch_size: int):
    ensure_output_dirs()
    set_seed(seed)
    arrays, meta = load_processed(processed_dir)
    edge_path = processed_dir / GRAPH_NAMES[graph_name]
    if not edge_path.exists():
        raise FileNotFoundError(f"Build the graph before training: {edge_path}")
    edge_index = torch.load(edge_path, map_location="cpu", weights_only=True)
    _, loaders = make_pyg_loaders(arrays, edge_index, batch_size, seed)
    cfg = TrainConfig()
    dims = embedding_dimensions(meta["category_sizes"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = NodeEncoder(len(meta["numeric_cols"]), meta["categorical_cols"],
                          meta["category_sizes"], dims,
                          hidden_dim=cfg.hidden_dim, dropout=cfg.encoder_dropout)
    model = GraphSAGEFraud(encoder, hidden_dim=cfg.hidden_dim,
                           dropout=cfg.classifier_dropout).to(device)
    optimizer, criterion = create_optimizer_and_loss(
        model, arrays["y"], arrays["train_mask"], device, cfg.lr, cfg.weight_decay
    )
    model, selection, history = fit_early_stopping(
        model, loaders, optimizer, criterion, device,
        max_epochs=epochs, patience=patience, graph=True
    )
    test = evaluate(model, loaders["test"], criterion, device, graph=True)
    results = {"model": "GraphSAGE", "graph": graph_name, "seed": seed,
               "num_edges": int(edge_index.shape[1]), "layers": 1,
               **selection, "test_pr_auc": test["pr_auc"],
               "test_roc_auc": test["roc_auc"], "test_loss": test["loss"]}
    name = f"graphsage_{graph_name}_seed{seed}"
    torch.save({"model_state_dict": model.state_dict(), "config": results,
                "embedding_dims": dims}, MODEL_DIR / f"{name}.pt")
    pd.DataFrame(history).to_csv(METRICS_DIR / f"{name}_history.csv", index=False)
    save_json(METRICS_DIR / f"{name}.json", results)
    print("Final GNN:", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--graph", choices=GRAPH_NAMES, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    main(args.processed_dir, args.graph, args.seed, args.max_epochs,
         args.patience, args.batch_size)
