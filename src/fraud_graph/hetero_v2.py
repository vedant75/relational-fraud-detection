"""Leakage-safe, typed historical context nodes for the IEEE-CIS fraud graph.

A context node is NOT a globally shared card/device entity. Each context node is
an immutable history snapshot belonging to exactly one destination transaction.
The only permitted length-two paths are:
    strictly earlier transaction -> historical context -> target transaction.
This prevents the future-to-past contamination of static bidirectional entity graphs.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

ADDR_HISTORY = "address_history"
DEVICE_HISTORY = "device_history"
TX = "transaction"
ADDR_IN = (TX, "prior_address_transaction", ADDR_HISTORY)
ADDR_OUT = (ADDR_HISTORY, "address_context_for", TX)
DEVICE_IN = (TX, "prior_device_transaction", DEVICE_HISTORY)
DEVICE_OUT = (DEVICE_HISTORY, "device_context_for", TX)
EDGE_TYPES = (ADDR_IN, ADDR_OUT, DEVICE_IN, DEVICE_OUT)


def context_edges(edge_index: torch.Tensor, num_nodes: int, k: int = 20):
    """Compact one snapshot per destination with >=1 predecessor.

    Returns (owner transaction IDs, predecessor->context edges, context->owner
    edges, normalized log of strictly historical incoming edge counts).
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("Expected edge_index shape [2, E]")
    if edge_index.dtype != torch.long:
        raise ValueError("edge_index must use torch.long")
    if k < 1 or num_nodes < 1:
        raise ValueError("k and num_nodes must be positive")
    if edge_index.numel() and (int(edge_index.min()) < 0 or int(edge_index.max()) >= num_nodes):
        raise ValueError("Edge node ID outside transaction range")

    src, dst = edge_index
    owners, inverse, counts = torch.unique(
        dst, sorted=True, return_inverse=True, return_counts=True
    )
    if counts.numel() and int(counts.max()) > k:
        raise ValueError("Historical in-degree exceeds chosen per-relation K")
    transaction_to_context = torch.stack((src.clone(), inverse), dim=0)
    context_to_transaction = torch.stack(
        (torch.arange(owners.numel(), dtype=torch.long), owners), dim=0
    )
    log_count = torch.log1p(counts.float()).unsqueeze(-1) / math.log1p(k)
    return owners, transaction_to_context, context_to_transaction, log_count


def make_hetero_structure(
    address_edges: torch.Tensor,
    device_edges: torch.Tensor,
    transaction_times: np.ndarray | torch.Tensor,
    k: int = 20,
    window_days: int = 30,
):
    """Construct PyG HeteroData topology only (no duplicate 1-GB feature matrix)."""
    from torch_geometric.data import HeteroData

    times = torch.as_tensor(np.asarray(transaction_times), dtype=torch.long)
    num_nodes = times.numel()
    if num_nodes == 0 or window_days <= 0:
        raise ValueError("Transaction time series must be nonempty; window must be positive")
    if not bool(torch.all(times[:-1] <= times[1:])):
        raise ValueError("Transaction times must be sorted chronologically")

    graph = HeteroData()
    graph[TX].num_nodes = num_nodes
    graph.graph_window_days = window_days
    graph.graph_k = k

    for name, incoming, outgoing, raw_edges in [
        (ADDR_HISTORY, ADDR_IN, ADDR_OUT, address_edges),
        (DEVICE_HISTORY, DEVICE_IN, DEVICE_OUT, device_edges),
    ]:
        src, dst = raw_edges
        if raw_edges.numel():
            if not bool(torch.all(times[src] < times[dst])):
                raise ValueError(f"{name}: equal-time/future transaction link detected")
            if not bool(torch.all(times[dst] - times[src] <= window_days * 86400)):
                raise ValueError(f"{name}: outside temporal window")
        owners, to_ctx, to_tx, log_count = context_edges(raw_edges, num_nodes, k)
        graph[name].num_nodes = owners.numel()
        graph[name].owner_node_id = owners
        graph[name].log_count = log_count
        graph[incoming].edge_index = to_ctx
        graph[outgoing].edge_index = to_tx

    validate_hetero_structure(graph, times, k, window_days)
    return graph


def validate_hetero_structure(graph, times, k: int = 20, window_days: int = 30) -> dict:
    """Verify every historic-tx -> context -> target path is strictly causal."""
    times = torch.as_tensor(np.asarray(times), dtype=torch.long)
    assert graph[TX].num_nodes == len(times)
    results = {"num_transactions": int(len(times))}
    for name, incoming, outgoing in [
        (ADDR_HISTORY, ADDR_IN, ADDR_OUT),
        (DEVICE_HISTORY, DEVICE_IN, DEVICE_OUT),
    ]:
        owners = graph[name].owner_node_id
        to_context = graph[incoming].edge_index
        back = graph[outgoing].edge_index
        count = graph[name].log_count
        assert graph[name].num_nodes == owners.numel()
        assert back.shape == (2, owners.numel())
        assert torch.equal(back[0], torch.arange(owners.numel()))
        assert torch.equal(back[1], owners)
        assert count.shape == (len(owners), 1)
        assert to_context.shape[0] == 2
        if to_context.numel():
            history_src, ctx_id = to_context
            dst_time = times[owners[ctx_id]]
            dt = dst_time - times[history_src]
            assert bool(torch.all(dt > 0)), f"{name} contains future leakage"
            assert bool(torch.all(dt <= window_days * 86400))
            degrees = torch.bincount(ctx_id, minlength=len(owners))
            assert bool(torch.all((degrees > 0) & (degrees <= k)))
            expected = torch.log1p(degrees.float()) / math.log1p(k)
            assert torch.allclose(count.squeeze(-1), expected, atol=1e-6)
        results[f"{name}_nodes"] = int(len(owners))
        results[f"{name}_history_edges"] = int(to_context.shape[1])
        results[f"{name}_membership_edges"] = int(back.shape[1])
    return results


def load_structure(path: Path):
    """Only load structures produced locally by our own builder, not untrusted .pt files."""
    return torch.load(path, map_location="cpu", weights_only=False)
