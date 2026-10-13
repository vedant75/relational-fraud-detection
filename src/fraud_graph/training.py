"""Train the MLP and sampled GNN on the same weighted BCE objective."""

import copy
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score


def positive_weight(y: np.ndarray, train_mask: np.ndarray) -> float:
    subset = y[train_mask]
    n_pos = float(subset.sum())
    if n_pos <= 0:
        raise ValueError("Training split has no positive examples")
    return (len(subset) - n_pos) / n_pos


def create_optimizer_and_loss(model, train_y, train_mask, device, lr=1e-3, weight_decay=1e-4):
    pos_weight = positive_weight(train_y, train_mask)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer, criterion


def batch_logits(model, batch, device, graph: bool):
    if graph:
        batch = batch.to(device)
        logits = model(batch.x_num.float(), batch.x_cat.long(), batch.edge_index)
        return logits[:batch.batch_size], batch.y[:batch.batch_size]
    numeric, categorical, labels = batch
    logits = model(numeric.to(device), categorical.to(device))
    return logits, labels.to(device)


def train_one_epoch(model, loader, optimizer, criterion, device, *, graph=False):
    model.train()
    total_loss = 0.0
    total_examples = 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        logits, labels = batch_logits(model, batch, device, graph)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        n = len(labels)
        total_loss += float(loss.item()) * n
        total_examples += n
    return total_loss / total_examples


@torch.no_grad()
def evaluate(model, loader, criterion, device, *, graph=False):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_probs = []
    all_labels = []
    for batch in loader:
        logits, labels = batch_logits(model, batch, device, graph)
        loss = criterion(logits, labels)
        n = len(labels)
        total_loss += float(loss.item()) * n
        total_examples += n
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    probabilities = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    return {
        "loss": total_loss / total_examples,
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
    }


def fit_early_stopping(model, loaders: dict, optimizer, criterion, device,
                       max_epochs: int = 15, patience: int = 3, *, graph=False):
    """Select checkpoint using validation sklearn Average Precision, never test."""
    best_score = -float("inf")
    best_epoch = 0
    no_improvement = 0
    best_state = None
    history = []
    for epoch in range(1, max_epochs + 1):
        start = time.perf_counter()
        train_loss = train_one_epoch(model, loaders["train"], optimizer, criterion,
                                     device, graph=graph)
        val = evaluate(model, loaders["val"], criterion, device, graph=graph)
        row = {"epoch": epoch, "train_loss": train_loss,
               "val_loss": val["loss"], "val_pr_auc": val["pr_auc"],
               "val_roc_auc": val["roc_auc"],
               "seconds": time.perf_counter() - start}
        history.append(row)
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
              f"val_loss={val['loss']:.4f} | val_PR_AUC={val['pr_auc']:.4f} | "
              f"val_ROC_AUC={val['roc_auc']:.4f} | time={row['seconds']:.1f}s",
              flush=True)
        if val["pr_auc"] > best_score:
            best_score = val["pr_auc"]
            best_epoch = epoch
            no_improvement = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print(f"Early stopping. Best epoch: {best_epoch}")
                break
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "val_pr_auc": best_score}, history
