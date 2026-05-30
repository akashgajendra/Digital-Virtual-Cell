import sys
from pathlib import Path

import pandas as pd


def latest_predictions_path() -> Path:
    paths = sorted(Path("runs").glob("*/predictions.csv"))
    if not paths:
        raise FileNotFoundError("No predictions.csv files found under runs/.")
    return paths[-1]


path = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_predictions_path()
df = pd.read_csv(path)

print(f"Reading: {path}")
print("First 50 rows:")
print(df.head(50))
print("\nShape:", df.shape)
print("\nInfo:")
print(df.info())
print("\nNumeric summary:")
print(df.describe())
print("\nCounts per compound:")
print(df.groupby("compound").size().describe())
