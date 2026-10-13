import numpy as np
import pandas as pd

from fraud_graph.preprocessing import (
    chronological_masks, encode_categories, encode_numeric, engineered_features, feature_schema,
)


def test_train_only_categorical_unknown_and_missing():
    frame = pd.DataFrame({"device": ["A", "B", None, "FUTURE", "A"]})
    train = np.array([True, True, True, False, False])
    encoded, maps, sizes = encode_categories(frame, ["device"], train)
    assert encoded[2, 0] == 0  # missing
    assert encoded[3, 0] == 1  # future category unknown, not learned
    assert encoded[0, 0] == encoded[4, 0] >= 2
    assert sizes["device"] == 4
    assert "FUTURE" not in maps["device"]


def test_numeric_bounds_and_median_use_train_only():
    frame = pd.DataFrame({"v": [0.0, 1.0, np.nan, 999999.0]})
    train = np.array([True, True, True, False])
    values, stats = encode_numeric(frame, ["v"], train)
    assert np.isfinite(values).all()
    assert stats["v"]["upper_bound"] < 10
    assert np.isclose(values[-1, 0], values[1, 0])
    assert np.isfinite(values[2, 0])


def test_temporal_split_does_not_split_ties():
    frame = pd.DataFrame({"TransactionDT": list(range(10))})
    masks, cutoffs = chronological_masks(frame)
    assert masks["train"].sum() == 7
    assert masks["val"].sum() == 1
    assert masks["test"].sum() == 2
    assert cutoffs == {"train_cutoff": 7, "val_cutoff": 8}


def test_safe_features_and_schema():
    frame = pd.DataFrame({
        "TransactionID": [1, 2], "TransactionDT": [1, 2], "isFraud": [0, 1],
        "DeviceInfo": ["iOS", None], "P_emaildomain": ["a", "a"],
        "R_emaildomain": ["a", "b"], "card1": [2.0, 3.0], "TransactionAmt": [10, 20],
    })
    result = engineered_features(frame)
    numeric, cat = feature_schema(result)
    assert "card1" in cat and "TransactionAmt" in numeric
    assert "isFraud" not in cat + numeric
    assert result["same_email_domain"].tolist() == [1, 0]
    assert result["has_device_info"].tolist() == [1, 0]
