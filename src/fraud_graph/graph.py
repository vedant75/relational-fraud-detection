"""Directed historical edges: strictly past transactions only.

Each target receives at most K latest predecessors with the same composite key
inside a W-day window. Equal-timestamp transactions never message one another.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .config import GraphConfig


def chronological_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Row i becomes node i. TransactionID is a deterministic tie breaker."""
    required = {"TransactionID", "TransactionDT"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(df.columns))}")
    if not df["TransactionID"].is_unique:
        raise ValueError("TransactionID must be unique")
    result = df.sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)
    result["node_id"] = np.arange(len(result), dtype=np.int64)
    return result


def build_temporal_edges(
    graph_df: pd.DataFrame,
    relation_cols: tuple[str, ...],
    cfg: GraphConfig = GraphConfig(),
) -> torch.Tensor:
    """Produce shape [2, E] with source time < destination time.

    graph_df must already have chronological row order and node_id = row index.
    Groups with missing relation fields remain isolated for this relation.
    """
    if cfg.window_days <= 0 or cfg.max_neighbors <= 0:
        raise ValueError("window_days and max_neighbors must be positive")
    if not relation_cols or not set(relation_cols).issubset(graph_df.columns):
        raise ValueError("relation_cols must exist in graph_df")
    if not np.array_equal(graph_df["node_id"].to_numpy(), np.arange(len(graph_df))):
        raise ValueError("node_id must match chronological dataframe row position")
    if not graph_df["TransactionDT"].is_monotonic_increasing:
        raise ValueError("Sort graph_df chronologically before building edges")

    eligible = graph_df.dropna(subset=list(relation_cols))
    window = cfg.window_days * 86400
    src_blocks: list[np.ndarray] = []
    dst_blocks: list[np.ndarray] = []

    for _, group in eligible.groupby(list(relation_cols), sort=False):
        if len(group) < 2:
            continue
        node_ids = group["node_id"].to_numpy(dtype=np.int64)
        times = group["TransactionDT"].to_numpy(dtype=np.int64)
        # Because graph_df is ordered by TransactionDT, every group is ordered too.
        for i, current_time in enumerate(times):
            left = np.searchsorted(times, current_time - window, side="left")
            right = np.searchsorted(times, current_time, side="left")
            start = max(left, right - cfg.max_neighbors)
            if right <= start:
                continue
            historical = node_ids[start:right]
            src_blocks.append(historical)
            dst_blocks.append(np.full(len(historical), node_ids[i], dtype=np.int64))

    if not src_blocks:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.from_numpy(np.stack([
        np.concatenate(src_blocks), np.concatenate(dst_blocks)
    ])).long()


def union_edges(*edges: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Deduplicated union; does not symmetrize or discard timestamps."""
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive")
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    encoded = []
    for edge in edges:
        if edge.ndim != 2 or edge.shape[0] != 2:
            raise ValueError("Each edge_index must have shape [2, E]")
        if edge.numel() and (edge.min() < 0 or edge.max() >= num_nodes):
            raise ValueError("Invalid node index")
        encoded.append(edge[0].long() * num_nodes + edge[1].long())
    ids = torch.unique(torch.cat(encoded))
    return torch.stack([ids // num_nodes, ids % num_nodes], dim=0)


def validate_temporal_edges(
    edge_index: torch.Tensor,
    graph_df: pd.DataFrame,
    cfg: GraphConfig | None = None,
    *,
    max_in_degree: int | None = None,
) -> dict:
    """Graph integrity; max_in_degree=None for combined graphs (can reach 2K)."""
    n = len(graph_df)
    assert edge_index.ndim == 2 and edge_index.shape[0] == 2
    assert edge_index.dtype == torch.long
    assert edge_index.numel() == 0 or (edge_index.min() >= 0 and edge_index.max() < n)
    src, dst = edge_index
    t = torch.as_tensor(graph_df["TransactionDT"].to_numpy(copy=True))
    assert bool(torch.all(t[src] < t[dst])), "Found future/equal-time source"
    if cfg is not None:
        assert bool(torch.all(t[dst] - t[src] <= cfg.window_days * 86400))
    indegree = torch.bincount(dst, minlength=n)
    if max_in_degree is not None:
        assert int(indegree.max()) <= max_in_degree
    return {
        "num_nodes": n,
        "num_edges": int(edge_index.shape[1]),
        "max_in_degree": int(indegree.max()),
        "mean_in_degree": float(indegree.float().mean()),
        "nodes_with_history": int((indegree > 0).sum()),
        "fraction_with_history": float((indegree > 0).float().mean()),
    }
