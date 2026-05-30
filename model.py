"""
Drug-seq transcriptomic response model.

Predicts per-gene log2 fold-change (L2FC) for a compound given its
molecular features, trained on pseudo-bulk aggregates from VCPI Drug-seq data.

Usage:
    uv run python model.py --train
    uv run python model.py --predict datasets/test_compounds.csv
    uv run python model.py --train --predict datasets/test_compounds.csv
"""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
import pickle

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASETS_DIR = Path("datasets")

# Physicochemical features available in the compounds CSV.
# Swap this list out for Morgan fingerprint columns if you add RDKit later.
COMPOUND_FEATURE_COLS = [
    "molecular_weight",
    "log_p",
    "tpsa",
    "num_rotatable_bonds",
    "num_h_acceptors",
    "num_h_donors",
    "num_atoms",
    "num_bonds",
]

N_TOP_GENES = 2000       # number of highly variable genes to model
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
HIDDEN_DIMS = [256, 512, 512, 256]
DROPOUT = 0.2
RUNS_DIR = Path("runs")


def make_run_dir() -> Path:
    """Create and return a new timestamped run directory under runs/."""
    from datetime import datetime
    run_dir = RUNS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def latest_run_dir() -> Path:
    """Return the most recently created run directory."""
    dirs = sorted(RUNS_DIR.glob("*"), key=lambda p: p.name)
    if not dirs:
        raise FileNotFoundError(
            f"No runs found in {RUNS_DIR}/. Run with --train first."
        )
    return dirs[-1]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _discover_file(pattern: str) -> Path:
    """Return the first file matching a glob pattern under DATASETS_DIR."""
    matches = sorted(glob.glob(str(DATASETS_DIR / pattern)))
    if not matches:
        raise FileNotFoundError(
            f"No file matching '{pattern}' found in {DATASETS_DIR}. "
            "Make sure your datasets folder contains the VCPI files."
        )
    return Path(matches[0])


def load_raw_data():
    """Load counts, metadata, and compound features from the datasets folder."""
    counts_path = _discover_file("vcpi_*_counts.parquet")
    meta_path = _discover_file("metadata-*.csv")
    compounds_path = _discover_file("compounds-*.csv")

    print(f"Loading counts from  : {counts_path}")
    print(f"Loading metadata from: {meta_path}")
    print(f"Loading compounds from: {compounds_path}")

    counts = pd.read_parquet(counts_path)           # (n_genes, n_cells+1)
    meta = pd.read_csv(meta_path)                   # (n_cells, 26)
    compounds_df = pd.read_csv(compounds_path)      # (n_compounds, 13)

    # counts has gene_id as the first column; make it the index
    counts = counts.set_index("gene_id")            # (n_genes, n_cells)
    counts.columns = counts.columns.astype(str)

    print(f"Counts shape : {counts.shape}  (genes × cells)")
    print(f"Metadata rows: {len(meta)}")
    print(f"Compounds    : {len(compounds_df)}")
    return counts, meta, compounds_df


def compute_pseudobulk_lfc(counts: pd.DataFrame, meta: pd.DataFrame):
    """
    Aggregate single-cell counts to pseudo-bulk per compound, normalize to
    log2-CPM, then compute L2FC relative to the DMSO control mean.

    Returns
    -------
    lfc_df : DataFrame  shape (n_compounds, n_genes)
    control_log2cpm : Series  shape (n_genes,)  — mean DMSO log2-CPM
    """
    meta = meta.copy()
    meta["sequenced_id"] = meta["sequenced_id"].astype(str)

    # Keep only cells present in the counts matrix
    cell_ids = counts.columns.tolist()
    meta = meta[meta["sequenced_id"].isin(cell_ids)].copy()

    # Separate controls (DMSO) from treated
    ctrl_mask = meta["is_control"].astype(bool)
    ctrl_ids = meta.loc[ctrl_mask, "sequenced_id"].tolist()
    trt_meta = meta[~ctrl_mask].copy()

    def _log2cpm(count_matrix: pd.DataFrame) -> pd.Series:
        """Sum across cells → CPM → log2(CPM+1)."""
        pseudobulk = count_matrix.sum(axis=1)          # sum over cells
        total = pseudobulk.sum()
        cpm = pseudobulk / total * 1e6
        return np.log2(cpm + 1)

    # Control baseline
    ctrl_log2cpm = _log2cpm(counts[ctrl_ids])

    # Per-compound pseudo-bulk L2FC
    rows = {}
    for compound_id, grp in trt_meta.groupby("compound"):
        cell_cols = [c for c in grp["sequenced_id"].tolist() if c in counts.columns]
        if not cell_cols:
            continue
        cmp_log2cpm = _log2cpm(counts[cell_cols])
        rows[compound_id] = cmp_log2cpm - ctrl_log2cpm   # L2FC in log space

    lfc_df = pd.DataFrame(rows).T   # (n_compounds, n_genes)
    lfc_df.index.name = "compound"
    print(f"Pseudo-bulk L2FC matrix: {lfc_df.shape}  (compounds × genes)")
    return lfc_df, ctrl_log2cpm


def select_top_genes(lfc_df: pd.DataFrame, n: int = N_TOP_GENES) -> list[str]:
    """Return the n genes with the highest variance in L2FC across compounds."""
    gene_var = lfc_df.var(axis=0)
    top = gene_var.nlargest(n).index.tolist()
    return top


# ---------------------------------------------------------------------------
# PyTorch dataset
# ---------------------------------------------------------------------------

class PerturbationDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.net(x))


class PerturbationMLP(nn.Module):
    """
    Maps compound molecular features → predicted per-gene L2FC.

    Architecture: input projection → residual MLP blocks → output projection.
    Hidden dims are fully configurable; residual blocks are inserted wherever
    consecutive hidden dims are equal.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int] = HIDDEN_DIMS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        layers = []

        # Input → first hidden
        layers += [
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]

        # Hidden → hidden (residual where dims match)
        for i in range(len(hidden_dims) - 1):
            d_in, d_out = hidden_dims[i], hidden_dims[i + 1]
            if d_in == d_out:
                layers.append(ResidualBlock(d_in, dropout))
            else:
                layers += [
                    nn.Linear(d_in, d_out),
                    nn.BatchNorm1d(d_out),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]

        # Final hidden → output (no activation — regression output)
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(epochs: int = EPOCHS, lr: float = LR) -> Path:
    """Full training pipeline. Creates a timestamped run dir and saves all artifacts there."""
    run_dir = make_run_dir()
    print(f"Run directory: {run_dir}\n")

    counts, meta, compounds_df = load_raw_data()

    print("\nComputing pseudo-bulk L2FC...")
    lfc_df, _ = compute_pseudobulk_lfc(counts, meta)

    print("Selecting top HVGs...")
    top_genes = select_top_genes(lfc_df, N_TOP_GENES)
    lfc_df = lfc_df[top_genes]

    # Align compound features with lfc rows
    compounds_df = compounds_df.set_index("compound")
    common = lfc_df.index.intersection(compounds_df.index)
    print(f"Compounds with both L2FC and features: {len(common)}")

    X_raw = compounds_df.loc[common, COMPOUND_FEATURE_COLS].values.astype(np.float32)
    y = lfc_df.loc[common].values.astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    dataset = PerturbationDataset(X, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = PerturbationMLP(input_dim=X.shape[1], output_dim=y.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(f"\nTraining on {len(dataset)} compounds, predicting {y.shape[1]} genes")
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        avg = total_loss / len(dataset)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{epochs}  MSE={avg:.4f}")

    checkpoint_path = run_dir / "model_checkpoint.pt"
    scaler_path = run_dir / "feature_scaler.pkl"
    gene_list_path = run_dir / "top_genes.txt"

    torch.save({"state_dict": model.state_dict(), "input_dim": X.shape[1], "output_dim": y.shape[1]}, checkpoint_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(gene_list_path, "w") as f:
        f.write("\n".join(top_genes))

    print(f"\nArtifacts saved to {run_dir}/")
    return run_dir


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def load_trained_model(run_dir: Path | None = None):
    """Load checkpoint, scaler, and gene list from run_dir (defaults to latest run)."""
    if run_dir is None:
        run_dir = latest_run_dir()
    print(f"Loading model from {run_dir}/")

    checkpoint_path = run_dir / "model_checkpoint.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint at {checkpoint_path}. Run with --train first.")

    ckpt = torch.load(checkpoint_path, weights_only=True)
    model = PerturbationMLP(input_dim=ckpt["input_dim"], output_dim=ckpt["output_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with open(run_dir / "feature_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(run_dir / "top_genes.txt") as f:
        top_genes = [line.strip() for line in f if line.strip()]

    return model, scaler, top_genes, run_dir


def predict(test_compounds_path: str, run_dir: Path | None = None) -> pd.DataFrame:
    """
    Predict L2FC for every (compound, gene) pair in test_compounds_path.

    Loads the model from run_dir (or the latest run if not specified).
    Writes predictions.parquet into the same run directory.

    Returns a DataFrame with columns [compound, gene_id, predicted_lfc].
    """
    model, scaler, top_genes, run_dir = load_trained_model(run_dir)

    test_df = pd.read_csv(test_compounds_path)
    print(f"Test compounds: {len(test_df)}")

    compounds_path = _discover_file("compounds-*.csv")
    compounds_df = pd.read_csv(compounds_path).set_index("compound")

    matched = test_df["compound"].isin(compounds_df.index)
    if not matched.all():
        print(f"Warning: {(~matched).sum()} test compounds not found in compounds file — skipping.")
    test_df = test_df[matched].copy()

    X_raw = compounds_df.loc[test_df["compound"], COMPOUND_FEATURE_COLS].values.astype(np.float32)
    X = scaler.transform(X_raw)

    with torch.no_grad():
        preds = model(torch.tensor(X)).numpy()

    records = [
        (cmpd, gene, float(preds[i, j]))
        for i, cmpd in enumerate(test_df["compound"].tolist())
        for j, gene in enumerate(top_genes)
    ]
    out = pd.DataFrame(records, columns=["compound", "gene_id", "predicted_lfc"])

    out_path = run_dir / "predictions.parquet"
    out.to_parquet(out_path, index=False)
    print(f"Wrote {out_path}  ({len(out):,} rows)")
    return out


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drug-seq perturbation model")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument(
        "--predict",
        metavar="TEST_CSV",
        help="Path to test_compounds.csv to generate predictions.parquet",
    )
    parser.add_argument(
        "--run-dir",
        metavar="DIR",
        help="Run directory to load model from (default: latest run)",
    )
    args = parser.parse_args()

    if not args.train and not args.predict:
        parser.print_help()

    run_dir = Path(args.run_dir) if args.run_dir else None

    if args.train:
        run_dir = train()
    if args.predict:
        predict(args.predict, run_dir=run_dir)
