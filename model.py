"""
Drug-seq transcriptomic response model.

Predicts per-gene perturbation response for a compound given its molecular
features, trained on pseudo-bulk aggregates from VCPI Drug-seq data. The
default target is absolute pseudo-bulk expression, log2(CPM+1); the old L2FC
target is available with --target-mode l2fc.

Usage:
    uv run python model.py --train
    uv run python model.py --predict datasets/test_compounds.csv
    uv run python model.py --train --predict datasets/test_compounds.csv
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
import pickle
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASETS_DIR = Path("datasets")

# Feature construction mode. Options: "fingerprints", "combined", "full".
FEATURE_MODE = "fingerprints"

# Prediction target. Options:
#   "log2cpm": absolute pseudo-bulk expression, log2(CPM+1)
#   "l2fc": log2 fold-change relative to DMSO controls
TARGET_MODE = "log2cpm"

# Physicochemical features available in the compounds CSV. Used by
# FEATURE_MODE="combined" and FEATURE_MODE="full".
COMPOUND_FEATURE_COLS = [
    "molecular_weight",
    "log_p",
    "tpsa",
    "num_rotatable_bonds",
    "num_h_acceptors",
    "num_h_donors",
    "num_atoms",
    "num_bonds",
    "purity_pct",
]

N_TOP_GENES = 2000       # number of highly variable genes to model
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
HIDDEN_DIMS = [256, 512, 512, 256]
DROPOUT = 0.2
VAL_SIZE = 0.2
RANDOM_STATE = 42
RUNS_DIR = Path("runs")

rdBase.DisableLog("rdApp.warning")


# ---------------------------------------------------------------------------
# Compound feature builders
# ---------------------------------------------------------------------------

def smiles_to_morgan(smiles, radius=2, nbits=2048):
    """Convert a SMILES string to a Morgan fingerprint bit vector."""
    if pd.isna(smiles):
        return np.zeros(nbits, dtype=np.float32)
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return np.zeros(nbits, dtype=np.float32)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    return np.array(fp, dtype=np.float32)


def smiles_to_physchem(smiles) -> dict:
    """Compute basic physicochemical descriptors from SMILES."""
    mol = Chem.MolFromSmiles(str(smiles)) if not pd.isna(smiles) else None
    if mol is None:
        return {col: 0.0 for col in COMPOUND_FEATURE_COLS}
    return {
        "molecular_weight": float(Descriptors.MolWt(mol)),
        "log_p": float(Descriptors.MolLogP(mol)),
        "tpsa": float(Descriptors.TPSA(mol)),
        "num_rotatable_bonds": float(Descriptors.NumRotatableBonds(mol)),
        "num_h_acceptors": float(Descriptors.NumHAcceptors(mol)),
        "num_h_donors": float(Descriptors.NumHDonors(mol)),
        "num_atoms": float(mol.GetNumAtoms()),
        "num_bonds": float(mol.GetNumBonds()),
        "purity_pct": 0.0,
    }


_tokenizer = None
_chemberta = None
_device = None


def _load_chemberta():
    """Lazy-load ChemBERTa only when FEATURE_MODE='full' needs it."""
    global _tokenizer, _chemberta, _device
    if _chemberta is None:
        print("Loading ChemBERTa model...")
        _device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        _tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
        _chemberta = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1").to(_device)
        _chemberta.eval()
        print(f"ChemBERTa loaded on {_device}")


def smiles_to_chemberta(smiles):
    """Convert a SMILES string to a ChemBERTa CLS embedding."""
    _load_chemberta()
    if pd.isna(smiles):
        smiles = ""
    tokens = _tokenizer(
        str(smiles),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    )
    tokens = {k: v.to(_device) for k, v in tokens.items()}
    with torch.no_grad():
        output = _chemberta(**tokens)
    return output.last_hidden_state[:, 0, :].squeeze().cpu().numpy()


_physchem_mean = None
_physchem_std = None


def _set_feature_state(state: dict | None):
    """Restore feature normalization state when predicting in a new process."""
    global _physchem_mean, _physchem_std
    if not state:
        return
    _physchem_mean = state.get("physchem_mean")
    _physchem_std = state.get("physchem_std")


def _get_feature_state() -> dict:
    """Capture feature normalization state so prediction can reuse train stats."""
    return {
        "feature_mode": FEATURE_MODE,
        "physchem_mean": _physchem_mean,
        "physchem_std": _physchem_std,
    }


def build_features(df, fit=False):
    """
    Build compound feature matrix according to FEATURE_MODE.

    fingerprints: Morgan fingerprints only.
    combined: Morgan fingerprints + normalized physicochemical descriptors.
    full: Morgan fingerprints + descriptors + ChemBERTa embeddings.
    """
    global _physchem_mean, _physchem_std

    if FEATURE_MODE not in ("fingerprints", "combined", "full"):
        raise ValueError(
            f"Unsupported FEATURE_MODE={FEATURE_MODE!r}. "
            "Use 'fingerprints', 'combined', or 'full'."
        )

    results = []

    if len(df) == 0:
        raise ValueError("No compounds available for feature construction.")

    if FEATURE_MODE in ("combined", "full"):
        missing = [col for col in COMPOUND_FEATURE_COLS if col not in df.columns]
        if missing:
            if "smiles" not in df.columns:
                raise ValueError(
                    "Missing physicochemical columns and no smiles column is available "
                    "to compute them."
                )
            computed = pd.DataFrame(df["smiles"].apply(smiles_to_physchem).tolist(), index=df.index)
            df = df.copy()
            for col in missing:
                df[col] = computed[col]

    if FEATURE_MODE in ("fingerprints", "combined", "full"):
        morgan = np.stack(df["smiles"].apply(smiles_to_morgan).values)
        results.append(morgan)

    if FEATURE_MODE in ("combined", "full"):
        physchem = (
            df[COMPOUND_FEATURE_COLS]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .values
        )
        if fit:
            _physchem_mean = physchem.mean(axis=0)
            _physchem_std = physchem.std(axis=0) + 1e-8
        if _physchem_mean is None or _physchem_std is None:
            raise RuntimeError(
                "Physicochemical normalization stats are missing. "
                "Call build_features(..., fit=True) first."
            )
        physchem = (physchem - _physchem_mean) / _physchem_std
        results.append(physchem.astype(np.float32))

    if FEATURE_MODE == "full":
        print(f"Computing ChemBERTa embeddings for {len(df)} compounds...")
        chemberta = np.stack(df["smiles"].apply(smiles_to_chemberta).values)
        results.append(chemberta.astype(np.float32))

    return np.hstack(results).astype(np.float32)


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


def compute_pseudobulk_targets(
    counts: pd.DataFrame,
    meta: pd.DataFrame,
    target_mode: str = TARGET_MODE,
):
    """
    Aggregate single-cell counts to pseudo-bulk per compound and normalize to
    log2-CPM. Depending on target_mode, return either absolute expression
    log2(CPM+1) or L2FC relative to the DMSO control mean.

    Returns
    -------
    target_df : DataFrame  shape (n_compounds, n_genes)
    control_log2cpm : Series  shape (n_genes,)  — mean DMSO log2-CPM
    """
    if target_mode not in ("log2cpm", "l2fc"):
        raise ValueError("target_mode must be 'log2cpm' or 'l2fc'")

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

    # Per-compound pseudo-bulk target
    rows = {}
    for compound_id, grp in trt_meta.groupby("compound"):
        cell_cols = [c for c in grp["sequenced_id"].tolist() if c in counts.columns]
        if not cell_cols:
            continue
        cmp_log2cpm = _log2cpm(counts[cell_cols])
        if target_mode == "l2fc":
            rows[compound_id] = cmp_log2cpm - ctrl_log2cpm
        else:
            rows[compound_id] = cmp_log2cpm

    target_df = pd.DataFrame(rows).T   # (n_compounds, n_genes)
    target_df.index.name = "compound"
    label = "L2FC" if target_mode == "l2fc" else "log2(CPM+1)"
    print(f"Pseudo-bulk {label} matrix: {target_df.shape}  (compounds × genes)")
    return target_df, ctrl_log2cpm


def compute_pseudobulk_lfc(counts: pd.DataFrame, meta: pd.DataFrame):
    """Backward-compatible wrapper for the original L2FC target."""
    return compute_pseudobulk_targets(counts, meta, target_mode="l2fc")


def select_top_genes(target_df: pd.DataFrame, n: int = N_TOP_GENES) -> list[str]:
    """Return the n genes with the highest variance across compounds."""
    gene_var = target_df.var(axis=0)
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


def evaluate_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return regression metrics on flattened compound-gene predictions."""
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    err = yp - yt
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    if np.std(yt) == 0 or np.std(yp) == 0:
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(yt, yp)[0, 1])
    return {"rmse": rmse, "mae": mae, "pearson": pearson}


def predict_array(model: nn.Module, X: np.ndarray) -> np.ndarray:
    """Run model inference for a feature matrix."""
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).numpy()


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
    Maps compound molecular features → predicted per-gene response target.

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

def train(
    epochs: int = EPOCHS,
    lr: float = LR,
    target_mode: str = TARGET_MODE,
    val_size: float = VAL_SIZE,
) -> Path:
    """Full training pipeline. Creates a timestamped run dir and saves all artifacts there."""
    run_dir = make_run_dir()
    print(f"Run directory: {run_dir}\n")

    counts, meta, compounds_df = load_raw_data()

    print(f"\nComputing pseudo-bulk targets ({target_mode})...")
    target_df, _ = compute_pseudobulk_targets(counts, meta, target_mode=target_mode)

    # Align compound features with lfc rows
    compounds_df = compounds_df.set_index("compound")
    common = target_df.index.intersection(compounds_df.index)
    print(f"Compounds with both targets and features: {len(common)}")

    if len(common) < 2:
        raise ValueError("Need at least two labeled compounds to create a validation split.")

    train_compounds, val_compounds = train_test_split(
        common.tolist(),
        test_size=val_size,
        random_state=RANDOM_STATE,
    )

    print("Selecting top HVGs from training compounds...")
    top_genes = select_top_genes(target_df.loc[train_compounds], N_TOP_GENES)
    target_df = target_df[top_genes]

    train_chem = compounds_df.loc[train_compounds].copy()
    val_chem = compounds_df.loc[val_compounds].copy()
    X_train = build_features(train_chem, fit=True)
    y_train = target_df.loc[train_compounds].values.astype(np.float32)
    X_val = build_features(val_chem, fit=False)
    y_val = target_df.loc[val_compounds].values.astype(np.float32)

    dataset = PerturbationDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = PerturbationMLP(input_dim=X_train.shape[1], output_dim=y_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(
        f"\nTraining on {len(dataset)} compounds, validating on {len(val_compounds)}, "
        f"predicting {y_train.shape[1]} genes"
    )
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

    val_pred = predict_array(model, X_val)
    val_metrics = evaluate_arrays(y_val, val_pred)
    print(
        "Validation: "
        f"RMSE={val_metrics['rmse']:.4f}  "
        f"MAE={val_metrics['mae']:.4f}  "
        f"Pearson={val_metrics['pearson']:.4f}"
    )

    checkpoint_path = run_dir / "model_checkpoint.pt"
    feature_state_path = run_dir / "feature_state.pkl"
    gene_list_path = run_dir / "top_genes.txt"
    metrics_path = run_dir / "validation_metrics.json"
    metadata_path = run_dir / "run_metadata.json"

    torch.save(
        {"state_dict": model.state_dict(), "input_dim": X_train.shape[1], "output_dim": y_train.shape[1]},
        checkpoint_path,
    )
    with open(feature_state_path, "wb") as f:
        pickle.dump(_get_feature_state(), f)
    with open(gene_list_path, "w") as f:
        f.write("\n".join(top_genes))
    with open(metrics_path, "w") as f:
        json.dump(val_metrics, f, indent=2)
    with open(metadata_path, "w") as f:
        json.dump(
            {
                "target_mode": target_mode,
                "feature_mode": FEATURE_MODE,
                "n_top_genes": N_TOP_GENES,
                "epochs": epochs,
                "lr": lr,
                "val_size": val_size,
                "random_state": RANDOM_STATE,
                "train_compounds": train_compounds,
                "val_compounds": val_compounds,
            },
            f,
            indent=2,
        )

    print(f"\nArtifacts saved to {run_dir}/")
    return run_dir


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def load_run_metadata(run_dir: Path) -> dict:
    """Load run metadata if present; older runs may not have it."""
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.exists():
        return {"target_mode": "l2fc"}
    with open(metadata_path) as f:
        return json.load(f)


def load_trained_model(run_dir: Path | None = None):
    """Load checkpoint, feature state, gene list, and metadata from run_dir."""
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

    feature_state_path = run_dir / "feature_state.pkl"
    if feature_state_path.exists():
        with open(feature_state_path, "rb") as f:
            _set_feature_state(pickle.load(f))
    with open(run_dir / "top_genes.txt") as f:
        top_genes = [line.strip() for line in f if line.strip()]
    metadata = load_run_metadata(run_dir)

    return model, top_genes, run_dir, metadata


def prepare_test_features(test_df: pd.DataFrame, compounds_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return feature rows in the same order as test_df while preserving official
    test compound IDs for output.
    """
    test_df = test_df.copy()
    test_ids = test_df["compound"].astype(str)

    compounds_by_uuid = compounds_df.copy()
    compounds_by_uuid.index = compounds_by_uuid.index.astype(str)
    uuid_matches = test_ids.isin(compounds_by_uuid.index)
    if uuid_matches.all():
        return compounds_by_uuid.loc[test_ids].reset_index(drop=True)

    if "user_compound_id" in compounds_df.columns:
        compounds_by_user_id = compounds_df.copy()
        compounds_by_user_id["user_compound_id"] = compounds_by_user_id["user_compound_id"].astype(str)
        compounds_by_user_id = compounds_by_user_id.drop_duplicates("user_compound_id").set_index("user_compound_id")
        user_id_matches = test_ids.isin(compounds_by_user_id.index)
        if user_id_matches.any():
            print(
                f"Matched {int(user_id_matches.sum())}/{len(test_df)} test compounds "
                "by user_compound_id."
            )
            matched_features = compounds_by_user_id.reindex(test_ids)
            fallback_cols = [col for col in test_df.columns if col not in matched_features.columns]
            for col in fallback_cols:
                matched_features[col] = test_df[col].values
            if matched_features["smiles"].isna().any() and "smiles" in test_df.columns:
                test_smiles = pd.Series(test_df["smiles"].values, index=matched_features.index)
                matched_features["smiles"] = matched_features["smiles"].combine_first(test_smiles)
            return matched_features.reset_index(drop=True)

    if "smiles" not in test_df.columns:
        raise ValueError(
            "No test compounds matched the training compounds file, and "
            "test_compounds.csv has no smiles column for feature generation."
        )

    print(
        "No test compound IDs matched the training compounds file. "
        "Using smiles from test_compounds.csv for feature generation."
    )
    return test_df.reset_index(drop=True)


def predict(test_compounds_path: str, run_dir: Path | None = None) -> pd.DataFrame:
    """
    Predict the trained target for every (compound, gene) pair in test_compounds_path.

    Loads the model from run_dir (or the latest run if not specified).
    Writes predictions.csv into the same run directory.

    Returns a DataFrame with columns [compound, gene_id, prediction] plus a
    target-specific prediction column for readability.
    """
    model, top_genes, run_dir, metadata = load_trained_model(run_dir)
    target_mode = metadata.get("target_mode", "l2fc")

    test_df = pd.read_csv(test_compounds_path)
    print(f"Test compounds: {len(test_df)}")

    compounds_path = _discover_file("compounds-*.csv")
    compounds_df = pd.read_csv(compounds_path).set_index("compound")
    test_chem = prepare_test_features(test_df, compounds_df)
    X = build_features(test_chem, fit=False)

    preds = predict_array(model, X)

    target_col = "predicted_lfc" if target_mode == "l2fc" else "predicted_log2cpm"

    records = [
        (cmpd, gene, float(preds[i, j]))
        for i, cmpd in enumerate(test_df["compound"].tolist())
        for j, gene in enumerate(top_genes)
    ]
    out = pd.DataFrame(records, columns=["compound", "gene_id", "prediction"])
    out[target_col] = out["prediction"]

    out_path = run_dir / "predictions.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}  ({len(out):,} rows)")
    return out


def compare_runs(old_run_dir: Path, new_run_dir: Path) -> dict:
    """Compare validation metrics from two trained runs."""
    old_metrics_path = old_run_dir / "validation_metrics.json"
    new_metrics_path = new_run_dir / "validation_metrics.json"
    if not old_metrics_path.exists():
        raise FileNotFoundError(f"No validation metrics found at {old_metrics_path}")
    if not new_metrics_path.exists():
        raise FileNotFoundError(f"No validation metrics found at {new_metrics_path}")

    with open(old_metrics_path) as f:
        old_metrics = json.load(f)
    with open(new_metrics_path) as f:
        new_metrics = json.load(f)

    old_meta = load_run_metadata(old_run_dir)
    new_meta = load_run_metadata(new_run_dir)

    comparison = {
        "old_run": str(old_run_dir),
        "new_run": str(new_run_dir),
        "old_target_mode": old_meta.get("target_mode", "l2fc"),
        "new_target_mode": new_meta.get("target_mode", "l2fc"),
        "metrics": {},
    }

    for metric in ("rmse", "mae", "pearson"):
        old_value = old_metrics.get(metric)
        new_value = new_metrics.get(metric)
        if old_value is None or new_value is None:
            continue
        if metric in ("rmse", "mae"):
            delta = old_value - new_value
            improved = new_value < old_value
            percent = delta / old_value * 100 if old_value else float("nan")
        else:
            delta = new_value - old_value
            improved = new_value > old_value
            percent = delta / abs(old_value) * 100 if old_value else float("nan")
        comparison["metrics"][metric] = {
            "old": old_value,
            "new": new_value,
            "delta": delta,
            "percent": percent,
            "improved": improved,
        }

    return comparison


def print_run_comparison(comparison: dict):
    """Print a human-readable run comparison."""
    print(f"Old run: {comparison['old_run']} ({comparison['old_target_mode']})")
    print(f"New run: {comparison['new_run']} ({comparison['new_target_mode']})")
    if comparison["old_target_mode"] != comparison["new_target_mode"]:
        print(
            "\nNote: target modes differ. Treat this as a held-out validation "
            "comparison for the intended output target, not as a comparison of "
            "test_compounds.csv predictions."
        )
    print("\nCriteria:")
    print("  RMSE/MAE improve when they decrease.")
    print("  Pearson improves when it increases.")
    print("\nValidation comparison:")
    for metric, values in comparison["metrics"].items():
        direction = "better" if values["improved"] else "worse"
        print(
            f"  {metric.upper():7s} old={values['old']:.6f} "
            f"new={values['new']:.6f} delta={values['delta']:.6f} "
            f"({values['percent']:.2f}%) {direction}"
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drug-seq perturbation model")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument(
        "--target-mode",
        choices=["log2cpm", "l2fc"],
        default=TARGET_MODE,
        help="Training target: absolute log2(CPM+1) expression or old DMSO-relative L2FC",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=VAL_SIZE,
        help="Fraction of labeled compounds held out for validation",
    )
    parser.add_argument(
        "--predict",
        metavar="TEST_CSV",
        help="Path to test_compounds.csv to generate predictions.csv",
    )
    parser.add_argument(
        "--run-dir",
        metavar="DIR",
        help="Run directory to load model from (default: latest run)",
    )
    parser.add_argument(
        "--compare-runs",
        nargs=2,
        metavar=("OLD_RUN", "NEW_RUN"),
        help="Compare validation metrics from two run directories",
    )
    args = parser.parse_args()

    if not args.train and not args.predict and not args.compare_runs:
        parser.print_help()

    if args.compare_runs:
        comparison = compare_runs(Path(args.compare_runs[0]), Path(args.compare_runs[1]))
        print_run_comparison(comparison)

    run_dir = Path(args.run_dir) if args.run_dir else None

    if args.train:
        run_dir = train(target_mode=args.target_mode, val_size=args.val_size)
    if args.predict:
        predict(args.predict, run_dir=run_dir)
