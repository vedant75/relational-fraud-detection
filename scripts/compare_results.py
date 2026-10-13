"""Print experiments saved by the training entrypoints."""

import json
from pathlib import Path

import pandas as pd

from fraud_graph.paths import METRICS_DIR


def main(metrics_dir: Path):
    rows = []
    for file in sorted(metrics_dir.glob("*.json")):
        if file.name.endswith("_history.json"):
            continue
        record = json.loads(file.read_text())
        if "test_pr_auc" in record:
            rows.append(record)
    if not rows:
        print(f"No model results in {metrics_dir}")
        return
    columns = ["model", "graph", "seed", "best_epoch", "best_iteration",
               "val_pr_auc", "test_pr_auc", "test_roc_auc"]
    print(pd.DataFrame(rows).reindex(columns=columns).to_string(index=False))


if __name__ == "__main__":
    main(METRICS_DIR)
