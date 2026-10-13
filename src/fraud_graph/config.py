"""Frozen V1 experiment settings (see README for observed benchmark results)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GraphConfig:
    window_days: int = 30
    max_neighbors: int = 20


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    batch_size: int = 1024
    hidden_dim: int = 128
    encoder_dropout: float = 0.1
    classifier_dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 15
    patience: int = 3
