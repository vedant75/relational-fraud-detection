"""Download IEEE-CIS training CSVs with the user's Kaggle credentials.

First visit https://www.kaggle.com/competitions/ieee-fraud-detection
and accept competition rules. Never commit raw data.
"""

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path

from fraud_graph.paths import RAW_DIR

FILES = ("train_transaction.csv", "train_identity.csv")


def download(destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    if not shutil.which("kaggle"):
        raise RuntimeError("Kaggle CLI is missing. Run `uv sync` first.")
    for filename in FILES:
        target = destination / filename
        archive = destination / (filename + ".zip")
        if target.exists():
            print("Already exists:", target)
            continue
        if not archive.exists():
            print("Downloading", filename, flush=True)
            subprocess.run([
                "kaggle", "competitions", "download", "-c", "ieee-fraud-detection",
                "-f", filename, "-p", str(destination)
            ], check=True)
        if not archive.exists():
            raise FileNotFoundError(f"Kaggle did not write expected archive: {archive}")
        with zipfile.ZipFile(archive) as zf:
            entries = [Path(name).name for name in zf.namelist()]
            if filename not in entries:
                raise ValueError(f"Missing {filename} inside {archive}")
            with zf.open(next(name for name in zf.namelist()
                              if Path(name).name == filename)) as src, target.open("wb") as out:
                shutil.copyfileobj(src, out)
        archive.unlink()
        print("Extracted:", target)
    print("Raw training files ready in", destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    download(parser.parse_args().raw_dir)
