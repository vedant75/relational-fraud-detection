import numpy as np
import pytest
import torch

from fraud_graph.hetero_v2 import context_edges


def test_context_edges_compact_mapping_and_log_counts():
    edges = torch.tensor([[0, 0, 1, 1], [1, 2, 2, 3]], dtype=torch.long)
    owners, historical, owned, log_count = context_edges(edges, num_nodes=4, k=20)
    assert owners.tolist() == [1, 2, 3]
    assert owned.tolist() == [[0, 1, 2], [1, 2, 3]]
    assert historical.tolist() == [[0, 0, 1, 1], [0, 1, 1, 2]]
    assert log_count.shape == (3, 1)


def test_reject_excess_degree():
    bad = torch.tensor([[0, 1, 2], [3, 3, 3]])
    with pytest.raises(ValueError, match="K"):
        context_edges(bad, num_nodes=4, k=2)


def test_causal_hetero_graph_and_future_rejection():
    pytest.importorskip("torch_geometric")
    from fraud_graph.hetero_v2 import (
        make_hetero_structure, validate_hetero_structure,
        ADDR_IN, ADDR_OUT, DEVICE_IN, DEVICE_OUT,
    )
    times = np.array([0, 86400, 86400 * 3, 86400 * 4])
    address = torch.tensor([[0, 0, 1], [1, 2, 2]])
    device = torch.tensor([[0, 1], [2, 3]])
    g = make_hetero_structure(address, device, times, k=3, window_days=30)
    stats = validate_hetero_structure(g, times, k=3, window_days=30)
    assert stats["address_history_nodes"] == 2
    assert stats["device_history_nodes"] == 2
    assert len(g.edge_types) == 4
    assert g[ADDR_IN].edge_index.shape == (2, 3)
    assert g[ADDR_OUT].edge_index.shape == (2, 2)
    assert g[DEVICE_IN].edge_index.shape == (2, 2)
    assert g[DEVICE_OUT].edge_index.shape == (2, 2)

    with pytest.raises(ValueError, match="future"):
        make_hetero_structure(torch.tensor([[3], [1]]), device, times, k=3)


def test_model_forward_and_gradient_when_pyg_installed():
    pytest.importorskip("torch_geometric")
    from fraud_graph.hetero_v2 import make_hetero_structure
    from fraud_graph.hetero_models import TemporalHeteroFraud
    from fraud_graph.models import NodeEncoder

    times = np.arange(6, dtype=np.int64) * 86400
    a = torch.tensor([[0, 0, 1, 2], [1, 2, 2, 3]], dtype=torch.long)
    d = torch.tensor([[0, 2], [4, 5]], dtype=torch.long)
    g = make_hetero_structure(a, d, times)
    g["transaction"].x_num = torch.randn(6, 3)
    g["transaction"].x_cat = torch.randint(0, 5, (6, 1))
    g["transaction"].y = torch.randint(0, 2, (6,)).float()
    model = TemporalHeteroFraud(
        NodeEncoder(3, ["cat"], {"cat": 5}, {"cat": 4}, hidden_dim=8), hidden_dim=8
    )
    logits = model(g)
    assert logits.shape == (6,)
    logits.sum().backward()
    assert model.addr_historical_mean.lin_l.weight.grad is not None
    assert torch.isfinite(logits).all()
