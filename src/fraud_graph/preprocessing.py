"""Point-in-time-safe feature preparation.

All vocabularies, winsorization cutoffs, medians, means and standard
 deviations are FIT only on the chronological training period.
"""

from __future__ import annotations

import gc
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import save_json

EXCLUDE = ["TransactionID", "isFraud", "TransactionDT"]
SEMANTIC_CATEGORICAL = [
    "ProductCD", *[f"card{i}" for i in range(1, 7)], "addr1", "addr2",
    "P_emaildomain", "R_emaildomain", *[f"M{i}" for i in range(1, 10)],
    "DeviceType", "DeviceInfo",
]


def engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Only present-transaction information; no label or future aggregates."""
    df = df.copy(deep=False)
    df["has_device_info"] = df["DeviceInfo"].notna().astype("int8")
    both = df["P_emaildomain"].notna() & df["R_emaildomain"].notna()
    df["both_email_domains_available"] = both.astype("int8")
    df["same_email_domain"] = (
        both & df["P_emaildomain"].eq(df["R_emaildomain"])
    ).astype("int8")
    return df


def feature_schema(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical = sorted(set(df.drop(columns=EXCLUDE).select_dtypes(
        include=["object", "string", "category"]
    ).columns) | (set(SEMANTIC_CATEGORICAL) & set(df.columns)))
    numeric = [c for c in df.columns if c not in categorical and c not in EXCLUDE]
    if not all(pd.api.types.is_numeric_dtype(df[c]) for c in numeric):
        raise TypeError("Some unclassified columns are not numeric")
    return numeric, categorical


def chronological_masks(df: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict]:
    if not df["TransactionDT"].is_monotonic_increasing:
        raise ValueError("Expected rows sorted by TransactionDT")
    times = df["TransactionDT"].to_numpy()
    n = len(df)
    # Equal timestamps always belong to the same split.
    train_cutoff = times[int(0.70 * n)]
    val_cutoff = times[int(0.85 * n)]
    masks = {
        "train": times < train_cutoff,
        "val": (times >= train_cutoff) & (times < val_cutoff),
        "test": times >= val_cutoff,
    }
    assert sum(int(m.sum()) for m in masks.values()) == n
    return masks, {"train_cutoff": int(train_cutoff), "val_cutoff": int(val_cutoff)}


def encode_categories(
    df: pd.DataFrame,
    categorical_cols: list[str],
    train_mask: np.ndarray,
    output_file: Path | None = None,
) -> tuple[np.ndarray, dict, dict]:
    """0=MISSING, 1=UNKNOWN at validation/inference, 2+=train-known.

    Map each column independently: integer identity is not numeric magnitude.
    """
    shape = (len(df), len(categorical_cols))
    result = (
        np.lib.format.open_memmap(output_file, mode="w+", dtype="int32", shape=shape)
        if output_file else np.empty(shape, dtype=np.int32)
    )
    mappings: dict[str, dict[str, int]] = {}
    sizes: dict[str, int] = {}
    for j, col in enumerate(categorical_cols):
        values = df[col].astype("string")
        unique_train = values.iloc[np.flatnonzero(train_mask)].dropna().unique().tolist()
        mapping = {str(value): idx + 2 for idx, value in enumerate(unique_train)}
        ids = values.map(mapping).fillna(1)
        ids.loc[values.isna()] = 0
        result[:, j] = ids.to_numpy(dtype=np.int32)
        mappings[col] = mapping
        sizes[col] = len(mapping) + 2
    if hasattr(result, "flush"):
        result.flush()
    return result, mappings, sizes


def encode_numeric(
    df: pd.DataFrame,
    numeric_cols: list[str],
    train_mask: np.ndarray,
    output_file: Path | None = None,
) -> tuple[np.ndarray, dict[str, dict[str, float]]]:
    """Columnwise float32 pipeline avoids several 2GB Pandas copies.

    Fit: train 0.1%/99.9% quantiles -> winsorize -> train median ->
    train population mean/std. Apply those train statistics to every split.
    """
    shape = (len(df), len(numeric_cols))
    result = (
        np.lib.format.open_memmap(output_file, mode="w+", dtype="float32", shape=shape)
        if output_file else np.empty(shape, dtype=np.float32)
    )
    stats: dict[str, dict[str, float]] = {}
    for j, col in enumerate(numeric_cols):
        values = df[col].to_numpy(dtype=np.float32, copy=True)
        train_values = values[train_mask]
        if not np.isfinite(train_values).any():
            raise ValueError(f"Numeric column entirely missing/nonfinite in train: {col}")
        # Infinity is treated as missing rather than influencing train statistics.
        values[~np.isfinite(values)] = np.nan
        train_values = values[train_mask]
        lower = float(np.nanquantile(train_values, 0.001))
        upper = float(np.nanquantile(train_values, 0.999))
        np.clip(values, lower, upper, out=values)
        median = float(np.nanmedian(values[train_mask]))
        values[np.isnan(values)] = median
        mean = float(values[train_mask].mean())
        std = float(values[train_mask].std(ddof=0))
        if std == 0 or not np.isfinite(std):
            std = 1.0
        result[:, j] = (values - mean) / std
        stats[col] = {"lower_bound": lower, "upper_bound": upper,
                      "median": median, "mean": mean, "std": std}
    if hasattr(result, "flush"):
        result.flush()
    return result, stats


def build_processed_data(model_df: pd.DataFrame, output_dir: Path) -> dict:
    """Sort, split, fit preprocessing, and persist small metadata + arrays."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if not model_df["TransactionID"].is_unique:
        raise ValueError("TransactionID must uniquely identify transactions")
    model_df = model_df.sort_values(["TransactionDT", "TransactionID"]).reset_index(drop=True)
    model_df = engineered_features(model_df)
    masks, cutoffs = chronological_masks(model_df)
    numeric_cols, categorical_cols = feature_schema(model_df)

    # Persist alignment key. Graph-building scripts verify it later.
    np.save(output_dir / "node_transaction_ids.npy",
            model_df["TransactionID"].to_numpy(dtype=np.int64))
    np.save(output_dir / "transaction_times.npy",
            model_df["TransactionDT"].to_numpy(dtype=np.int64))
    np.save(output_dir / "y.npy", model_df["isFraud"].to_numpy(dtype=np.float32))
    for name, mask in masks.items():
        np.save(output_dir / f"{name}_mask.npy", mask.astype(bool))

    cat, category_maps, category_sizes = encode_categories(
        model_df, categorical_cols, masks["train"], output_dir / "categorical_features.npy"
    )
    cat_shape = list(cat.shape)
    del cat
    gc.collect()
    num, stats = encode_numeric(
        model_df, numeric_cols, masks["train"], output_dir / "numeric_features.npy"
    )
    num_shape = list(num.shape)
    del num
    gc.collect()

    metadata = {
        "numeric_cols": numeric_cols, "categorical_cols": categorical_cols,
        "category_sizes": category_sizes, "category_maps": category_maps,
        "numeric_stats": stats, "cutoffs": cutoffs,
        "numeric_shape": num_shape, "categorical_shape": cat_shape,
        "fraud_rate_by_split": {
            s: float(model_df.loc[mask, "isFraud"].mean()) for s, mask in masks.items()
        },
        "split_sizes": {s: int(mask.sum()) for s, mask in masks.items()},
    }
    save_json(output_dir / "preprocessing.json", metadata)
    return metadata
