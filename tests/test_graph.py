import pandas as pd
import torch

from fraud_graph.config import GraphConfig
from fraud_graph.graph import (
    chronological_frame, build_temporal_edges, union_edges, validate_temporal_edges,
)


def example():
    # Two transactions at t=100 intentionally share an unknown ordering.
    return chronological_frame(pd.DataFrame({
        "TransactionID": [105, 101, 104, 103, 102, 106],
        "TransactionDT": [500, 100, 300, 100, 200, 400],
        "card1": [1, 1, 1, 1, 1, 2],
        "addr1": [4, 4, 4, 4, 4, 4],
        "DeviceInfo": ["A", "A", "B", "A", "B", "C"],
    }))


def test_chronological_node_identity():
    df = example()
    assert df.TransactionID.tolist() == [101, 103, 102, 104, 106, 105]
    assert df.node_id.tolist() == list(range(6))


def test_strict_time_and_last_k():
    df = example()
    cfg = GraphConfig(window_days=30, max_neighbors=2)
    edges = build_temporal_edges(df, ("card1", "addr1"), cfg)
    src, dst = edges.tolist()
    actual = set(zip(src, dst))
    # For index 2 at 200: both 100-timestamp nodes are valid.
    assert (0, 2) in actual and (1, 2) in actual
    # For index 3 at 300: retain last two (the 200 node, one 100 node).
    assert (2, 3) in actual
    assert sum(d == 3 for d in dst) == 2
    # Same-timestamp transactions cannot communicate.
    assert (0, 1) not in actual and (1, 0) not in actual
    summary = validate_temporal_edges(edges, df, cfg, max_in_degree=2)
    assert summary["max_in_degree"] == 2


def test_time_window_excludes_stale():
    df = chronological_frame(pd.DataFrame({
        "TransactionID": [1, 2, 3],
        "TransactionDT": [0, 5 * 86400, 50 * 86400],
        "card1": [1, 1, 1],
    }))
    edges = build_temporal_edges(df, ("card1",), GraphConfig(30, 20))
    assert list(zip(*edges.tolist())) == [(0, 1)]


def test_relation_and_missingness():
    df = chronological_frame(pd.DataFrame({
        "TransactionID": [1, 2, 3, 4],
        "TransactionDT": [10, 20, 30, 40],
        "card1": [1, 1, 1, 1],
        "DeviceInfo": ["A", "B", "A", None],
    }))
    edges = build_temporal_edges(df, ("card1", "DeviceInfo"))
    assert list(zip(*edges.tolist())) == [(0, 2)]


def test_union_deduplicates_and_keeps_direction():
    a = torch.tensor([[0, 1], [2, 2]], dtype=torch.long)
    b = torch.tensor([[0, 3], [2, 4]], dtype=torch.long)
    united = union_edges(a, b, num_nodes=5)
    assert set(zip(*united.tolist())) == {(0, 2), (1, 2), (3, 4)}
