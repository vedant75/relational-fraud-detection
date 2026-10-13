import torch

from fraud_graph.models import FraudMLP, NodeEncoder, embedding_dimensions


def test_encoder_and_mlp_shapes_and_gradients():
    sizes = {"card1": 7, "device": 5}
    dims = embedding_dimensions(sizes)
    encoder = NodeEncoder(3, ["card1", "device"], sizes, dims, hidden_dim=8)
    model = FraudMLP(encoder, hidden_dim=8)
    x_num = torch.randn(4, 3)
    x_cat = torch.tensor([[0, 1], [2, 3], [4, 0], [6, 4]], dtype=torch.long)
    logits = model(x_num, x_cat)
    assert logits.shape == (4,)
    logits.sum().backward()
    assert model.encoder.embeddings[0].weight.grad is not None


def test_gnn_if_pyg_installed():
    import pytest
    pytest.importorskip("torch_geometric")
    from fraud_graph.models import GraphSAGEFraud

    sizes = {"cat": 4}
    encoder = NodeEncoder(2, ["cat"], sizes, {"cat": 4}, hidden_dim=8)
    model = GraphSAGEFraud(encoder, hidden_dim=8)
    num = torch.randn(3, 2)
    cat = torch.tensor([[0], [2], [3]])
    edge_index = torch.tensor([[0, 1], [2, 2]])
    logits = model(num, cat, edge_index)
    assert logits.shape == (3,)
    logits.sum().backward()
    assert model.sage.lin_l.weight.grad is not None
