"""Read processed features without recreating multi-gigabyte Pandas objects."""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .paths import PROCESSED_DIR


class FraudDataset(Dataset):
    """Independent transactions for the no-graph MLP baseline."""

    def __init__(self, arrays: dict, indices: np.ndarray):
        self.arrays = arrays
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row_id = int(self.indices[index])
        num = np.array(self.arrays["numeric"][row_id], dtype=np.float32, copy=True)
        cat = np.array(self.arrays["categorical"][row_id], dtype=np.int64, copy=True)
        label = np.float32(self.arrays["y"][row_id])
        return torch.from_numpy(num), torch.from_numpy(cat), torch.tensor(label)


def load_processed(processed_dir: Path = PROCESSED_DIR) -> tuple[dict, dict]:
    metadata = json.loads((processed_dir / "preprocessing.json").read_text())
    arrays = {
        "numeric": np.load(processed_dir / "numeric_features.npy", mmap_mode="c"),
        "categorical": np.load(processed_dir / "categorical_features.npy", mmap_mode="c"),
        "y": np.load(processed_dir / "y.npy"),
        "train_mask": np.load(processed_dir / "train_mask.npy"),
        "val_mask": np.load(processed_dir / "val_mask.npy"),
        "test_mask": np.load(processed_dir / "test_mask.npy"),
    }
    n = len(arrays["y"])
    assert arrays["numeric"].shape == (n, len(metadata["numeric_cols"]))
    assert arrays["categorical"].shape == (n, len(metadata["categorical_cols"]))
    assert sum(int(arrays[f"{s}_mask"].sum()) for s in ("train", "val", "test")) == n
    return arrays, metadata


def make_mlp_loaders(arrays: dict, batch_size: int, seed: int) -> dict:
    generator = torch.Generator().manual_seed(seed)
    result = {}
    for name in ("train", "val", "test"):
        indices = np.flatnonzero(arrays[f"{name}_mask"])
        result[name] = DataLoader(
            FraudDataset(arrays, indices), batch_size=batch_size,
            shuffle=name == "train", num_workers=0,
            generator=generator if name == "train" else None,
        )
    return result


def make_pyg_loaders(arrays: dict, edge_index: torch.Tensor, batch_size: int, seed: int):
    from torch_geometric.data import Data
    from torch_geometric.loader import NeighborLoader

    num_nodes = len(arrays["y"])
    graph = Data(
        x_num=torch.from_numpy(arrays["numeric"]),
        x_cat=torch.from_numpy(arrays["categorical"]),
        y=torch.from_numpy(arrays["y"]).float(),
        edge_index=edge_index,
        train_mask=torch.from_numpy(arrays["train_mask"]),
        val_mask=torch.from_numpy(arrays["val_mask"]),
        test_mask=torch.from_numpy(arrays["test_mask"]),
        num_nodes=num_nodes,
    )
    loaders = {}
    for name in ("train", "val", "test"):
        loaders[name] = NeighborLoader(
            graph, input_nodes=getattr(graph, f"{name}_mask"), num_neighbors=[-1],
            batch_size=batch_size, shuffle=name == "train", num_workers=0,
            generator=torch.Generator().manual_seed(seed) if name == "train" else None,
        )
    return graph, loaders
