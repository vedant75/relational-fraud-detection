"""Node-only baseline: no graph, same feature encoder as GraphSAGE."""

import argparse
from pathlib import Path

import pandas as pd
import torch

from fraud_graph.config import TrainConfig
from fraud_graph.data import load_processed, make_mlp_loaders
from fraud_graph.models import NodeEncoder, FraudMLP, embedding_dimensions
from fraud_graph.paths import MODEL_DIR, METRICS_DIR, PROCESSED_DIR, ensure_output_dirs
from fraud_graph.training import create_optimizer_and_loss, evaluate, fit_early_stopping
from fraud_graph.utils import save_json, set_seed


def main(processed_dir: Path, seed: int, epochs: int, patience: int, batch_size: int):
    ensure_output_dirs()
    set_seed(seed)
    arrays, meta = load_processed(processed_dir)
    dims = embedding_dimensions(meta["category_sizes"])
    loaders = make_mlp_loaders(arrays, batch_size, seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = TrainConfig()
    encoder = NodeEncoder(len(meta["numeric_cols"]), meta["categorical_cols"],
                          meta["category_sizes"], dims,
                          hidden_dim=cfg.hidden_dim, dropout=cfg.encoder_dropout)
    model = FraudMLP(encoder, hidden_dim=cfg.hidden_dim,
                     dropout=cfg.classifier_dropout).to(device)
    optimizer, criterion = create_optimizer_and_loss(
        model, arrays["y"], arrays["train_mask"], device, cfg.lr, cfg.weight_decay
    )
    model, selection, history = fit_early_stopping(
        model, loaders, optimizer, criterion, device, max_epochs=epochs, patience=patience
    )
    # A SINGLE terminal test evaluation: do not tune on this result.
    test = evaluate(model, loaders["test"], criterion, device)
    results = {"model": "MLP", "graph": None, "seed": seed, **selection,
               "test_pr_auc": test["pr_auc"], "test_roc_auc": test["roc_auc"],
               "test_loss": test["loss"]}
    torch.save({"model_state_dict": model.state_dict(), "config": results,
                "embedding_dims": dims}, MODEL_DIR / f"mlp_seed{seed}.pt")
    pd.DataFrame(history).to_csv(METRICS_DIR / f"mlp_seed{seed}_history.csv", index=False)
    save_json(METRICS_DIR / f"mlp_seed{seed}.json", results)
    print("Final MLP:", results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    main(args.processed_dir, args.seed, args.max_epochs, args.patience, args.batch_size)
