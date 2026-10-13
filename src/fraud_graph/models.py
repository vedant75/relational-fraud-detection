"""Shared mixed-type node encoder, tabular MLP, one-layer GraphSAGE."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def embedding_dimensions(category_sizes: dict[str, int]) -> dict[str, int]:
    return {name: min(32, max(4, math.ceil(math.sqrt(size))))
            for name, size in category_sizes.items()}


class NodeEncoder(nn.Module):
    def __init__(self, numeric_dim: int, categorical_cols: list[str],
                 category_sizes: dict[str, int], embedding_dims: dict[str, int],
                 hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(category_sizes[col], embedding_dims[col])
            for col in categorical_cols
        ])
        self.numeric_encoder = nn.Sequential(
            nn.Linear(numeric_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + sum(embedding_dims[col] for col in categorical_cols),
                      hidden_dim),
            nn.ReLU(), nn.Dropout(dropout)
        )

    def forward(self, numeric_x: torch.Tensor, categorical_x: torch.Tensor) -> torch.Tensor:
        numeric_h = self.numeric_encoder(numeric_x)
        categorical_h = torch.cat([
            embedding(categorical_x[:, j]) for j, embedding in enumerate(self.embeddings)
        ], dim=1)
        return self.fusion(torch.cat([numeric_h, categorical_h], dim=1))


class FraudMLP(nn.Module):
    def __init__(self, encoder: NodeEncoder, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )

    def forward(self, numeric_x: torch.Tensor, categorical_x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(numeric_x, categorical_x)).squeeze(-1)


class GraphSAGEFraud(nn.Module):
    def __init__(self, encoder: NodeEncoder, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        try:
            from torch_geometric.nn import SAGEConv
        except ImportError as exc:
            raise ImportError("GraphSAGE requires torch-geometric") from exc
        self.encoder = encoder
        self.sage = SAGEConv(hidden_dim, hidden_dim)  # mean aggregator, as in original
        self.dropout = dropout
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1)
        )

    def forward(self, numeric_x: torch.Tensor, categorical_x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        h = self.encoder(numeric_x, categorical_x)
        h = F.relu(self.sage(h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.classifier(h).squeeze(-1)
