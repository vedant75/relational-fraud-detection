"""Two-stage, relation-aware temporal heterogeneous fraud classifier.

Stage 1: historical transaction embeddings -> typed history snapshot.
Stage 2: typed history snapshot -> transaction fraud logit.
No future edges are present in the supplied causal snapshot graph.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hetero_v2 import ADDR_IN, ADDR_OUT, ADDR_HISTORY, DEVICE_IN, DEVICE_OUT, DEVICE_HISTORY, TX


class TemporalHeteroFraud(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        from torch_geometric.nn import SAGEConv

        self.encoder = encoder
        self.addr_historical_mean = SAGEConv((hidden_dim, hidden_dim), hidden_dim, root_weight=False)
        self.device_historical_mean = SAGEConv((hidden_dim, hidden_dim), hidden_dim, root_weight=False)
        self.addr_count_encoder = nn.Linear(1, hidden_dim, bias=False)
        self.device_count_encoder = nn.Linear(1, hidden_dim, bias=False)
        self.addr_to_target = SAGEConv((hidden_dim, hidden_dim), hidden_dim, root_weight=False)
        self.device_to_target = SAGEConv((hidden_dim, hidden_dim), hidden_dim, root_weight=False)
        self.classifier = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
        )
        self.dropout = dropout

    def _history_signal(self, batch, tx_h, node_type, in_type, out_type, hist_conv, count_layer, out_conv):
        n_context = batch[node_type].num_nodes
        n_tx = tx_h.size(0)
        if n_context == 0:
            return tx_h.new_zeros((n_tx, tx_h.size(1)))

        # No persistent/global entity embedding: aggregation uses only historical
        # transaction messages. No center-node self-message in stage 1.
        h_history = hist_conv(
            (tx_h, None), batch[in_type].edge_index,
            size=(n_tx, n_context),
        )
        h_history = F.relu(
            h_history + count_layer(batch[node_type].log_count.float())
        )
        h_history = F.dropout(h_history, p=self.dropout, training=self.training)
        # Each snapshot has exactly one outbound edge to its owner transaction.
        return out_conv(
            (h_history, None), batch[out_type].edge_index,
            size=(n_context, n_tx),
        )

    def forward(self, batch):
        tx = batch[TX]
        tx_h = self.encoder(tx.x_num.float(), tx.x_cat.long())
        addr = self._history_signal(
            batch, tx_h, ADDR_HISTORY, ADDR_IN, ADDR_OUT,
            self.addr_historical_mean, self.addr_count_encoder, self.addr_to_target,
        )
        device = self._history_signal(
            batch, tx_h, DEVICE_HISTORY, DEVICE_IN, DEVICE_OUT,
            self.device_historical_mean, self.device_count_encoder, self.device_to_target,
        )
        combined = torch.cat([tx_h, F.relu(addr), F.relu(device)], dim=-1)
        return self.classifier(combined).squeeze(-1)
