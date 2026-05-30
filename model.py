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
from vcpi_prediction_contest import (
    aggregate_leaderboards,
    load_gene_filter,
    load_weights_matrix,
    score_compounds,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASETS_DIR = Path("datasets")

# Feature construction mode. The proposed full encoder is the default:
# Morgan(2048) + ChemBERTa(384) + UniMol(512) + physicochemical(9) = 2953.
FEATURE_MODE = "full"
MORGAN_DIM = 2048
CHEMBERTA_DIM = 384
UNIMOL_DIM = 512
COMPOUND_EMBED_DIM = 1024
TARGET_BRANCH_DIM = 64

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
MOLECULAR_INPUT_DIM = MORGAN_DIM + CHEMBERTA_DIM + UNIMOL_DIM + len(COMPOUND_FEATURE_COLS)

N_TOP_GENES = 2000       # number of highly variable genes to model
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
HIDDEN_DIMS = [1024, 512, 512, 256]
DROPOUT = 0.1
VAL_SIZE = 0.2
RANDOM_STATE = 42
RUNS_DIR = Path("runs")

rdBase.DisableLog("rdApp.warning")


# ---------------------------------------------------------------------------
# Compound feature builders
# ---------------------------------------------------------------------------

def smiles_to_morgan(smiles, radius=2, nbits=MORGAN_DIM):
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
_chembl_target_cols = []
_unimol_cache = None
_warned_missing_unimol = False


def _set_feature_state(state: dict | None):
    """Restore feature normalization state when predicting in a new process."""
    global _physchem_mean, _physchem_std, _chembl_target_cols
    if not state:
        return
    _physchem_mean = state.get("physchem_mean")
    _physchem_std = state.get("physchem_std")
    _chembl_target_cols = state.get("chembl_target_cols", [])


def _get_feature_state() -> dict:
    """Capture feature normalization state so prediction can reuse train stats."""
    return {
        "feature_mode": FEATURE_MODE,
        "physchem_mean": _physchem_mean,
        "physchem_std": _physchem_std,
        "chembl_target_cols": _chembl_target_cols,
    }


def _fixed_dim(vec, dim: int) -> np.ndarray:
    """Trim or zero-pad an embedding to the configured dimension."""
    out = np.zeros(dim, dtype=np.float32)
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    out[: min(dim, len(arr))] = arr[:dim]
    return out


def _ensure_physchem(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing physicochemical descriptor columns from SMILES when needed."""
    missing = [col for col in COMPOUND_FEATURE_COLS if col not in df.columns]
    if not missing:
        return df
    if "smiles" not in df.columns:
        raise ValueError(
            "Missing physicochemical columns and no smiles column is available "
            "to compute them."
        )
    computed = pd.DataFrame(df["smiles"].apply(smiles_to_physchem).tolist(), index=df.index)
    df = df.copy()
    for col in missing:
        df[col] = computed[col]
    return df


def _normalized_physchem(df: pd.DataFrame, fit: bool) -> np.ndarray:
    """Return normalized physicochemical descriptors."""
    global _physchem_mean, _physchem_std

    df = _ensure_physchem(df)
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
            "Call build_compound_inputs(..., fit=True) first."
        )
    return ((physchem - _physchem_mean) / _physchem_std).astype(np.float32)


def _load_unimol_cache() -> pd.DataFrame | None:
    """
    Load precomputed UniMol embeddings when present.

    Expected file: datasets/unimol_embeddings.csv, keyed by one of
    compound/user_compound_id/inchi_key/smiles, with 512 numeric embedding
    columns such as unimol_0 ... unimol_511.
    """
    global _unimol_cache, _warned_missing_unimol
    if _unimol_cache is not None:
        return _unimol_cache

    path = DATASETS_DIR / "unimol_embeddings.csv"
    if not path.exists():
        if not _warned_missing_unimol:
            print(
                "Warning: datasets/unimol_embeddings.csv not found; "
                "using zero UniMol embeddings."
            )
            _warned_missing_unimol = True
        return None

    cache = pd.read_csv(path)
    embed_cols = [col for col in cache.columns if col.startswith("unimol_")]
    if len(embed_cols) < UNIMOL_DIM:
        raise ValueError(
            f"{path} must contain at least {UNIMOL_DIM} columns named unimol_*."
        )
    _unimol_cache = cache
    return _unimol_cache


def _unimol_embeddings(df: pd.DataFrame) -> np.ndarray:
    """Return 512-dim UniMol embeddings, using zeros where unavailable."""
    cache = _load_unimol_cache()
    if cache is None:
        return np.zeros((len(df), UNIMOL_DIM), dtype=np.float32)

    embed_cols = [col for col in cache.columns if col.startswith("unimol_")][:UNIMOL_DIM]
    key_cols = ["compound", "user_compound_id", "inchi_key", "smiles"]
    for key in key_cols:
        if key in df.columns and key in cache.columns:
            lookup = cache.drop_duplicates(key).set_index(key)[embed_cols]
            matched = lookup.reindex(df[key])
            return matched.fillna(0).to_numpy(dtype=np.float32)
    return np.zeros((len(df), UNIMOL_DIM), dtype=np.float32)


def _chembl_target_columns(df: pd.DataFrame, fit: bool) -> list[str]:
    """Detect optional ChEMBL target columns and persist the training schema."""
    global _chembl_target_cols
    candidates = [
        col for col in df.columns
        if col.startswith("chembl_target_") or col.startswith("chembl_")
    ]
    if fit:
        _chembl_target_cols = sorted(candidates)
    return _chembl_target_cols


def _chembl_targets(df: pd.DataFrame, fit: bool) -> np.ndarray | None:
    """Return optional ChEMBL target features or None when not available."""
    cols = _chembl_target_columns(df, fit)
    if not cols:
        return None
    target_df = df.reindex(columns=cols, fill_value=0)
    return (
        target_df
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .to_numpy(dtype=np.float32)
    )


def build_compound_inputs(df, fit=False) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Build the proposed molecular input and optional ChEMBL target input.

    Molecular order:
    Morgan(2048), ChemBERTa(384), UniMol(512), physicochemical(9).
    """
    if len(df) == 0:
        raise ValueError("No compounds available for feature construction.")
    if "smiles" not in df.columns:
        raise ValueError("Compound features require a smiles column.")

    if FEATURE_MODE != "full":
        raise ValueError("The proposed architecture requires FEATURE_MODE='full'.")

    morgan = np.stack(df["smiles"].apply(smiles_to_morgan).values)

    print(f"Computing ChemBERTa embeddings for {len(df)} compounds...")
    chemberta = np.stack(
        df["smiles"].apply(lambda smiles: _fixed_dim(smiles_to_chemberta(smiles), CHEMBERTA_DIM)).values
    )

    unimol = _unimol_embeddings(df)
    physchem = _normalized_physchem(df, fit)
    molecular = np.hstack([morgan, chemberta, unimol, physchem]).astype(np.float32)
    if molecular.shape[1] != MOLECULAR_INPUT_DIM:
        raise RuntimeError(
            f"Expected molecular input dim {MOLECULAR_INPUT_DIM}, got {molecular.shape[1]}."
        )
    return molecular, _chembl_targets(df, fit)


def build_features(df, fit=False):
    """Backward-compatible wrapper returning only the molecular features."""
    molecular, _ = build_compound_inputs(df, fit=fit)
    return molecular


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


def _discover_files(pattern: str) -> list[Path]:
    """Return all files matching a glob pattern under DATASETS_DIR."""
    matches = sorted(glob.glob(str(DATASETS_DIR / pattern)))
    if not matches:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found in {DATASETS_DIR}. "
            "Make sure your datasets folder contains the VCPI files."
        )
    return [Path(match) for match in matches]


def load_raw_data():
    """Load and merge all counts, metadata, and compound files from datasets/."""
    counts_paths = _discover_files("vcpi_*_counts.parquet")
    meta_paths = _discover_files("metadata-*.csv")
    compounds_paths = _discover_files("compounds-*.csv")

    print("Loading counts from:")
    for path in counts_paths:
        print(f"  {path}")
    print("Loading metadata from:")
    for path in meta_paths:
        print(f"  {path}")
    print("Loading compounds from:")
    for path in compounds_paths:
        print(f"  {path}")

    counts_pieces = []
    for path in counts_paths:
        piece = pd.read_parquet(path).set_index("gene_id")
        counts_pieces.append(piece)
    counts = (
        pd.concat(counts_pieces, axis=1, join="outer")
        .fillna(0)
        .astype("int32")
    )
    counts.columns = counts.columns.astype(str)

    meta = pd.concat(
        [pd.read_csv(path) for path in meta_paths],
        ignore_index=True,
    )
    compounds_df = (
        pd.concat([pd.read_csv(path) for path in compounds_paths], ignore_index=True)
        .drop_duplicates(subset=["compound"])
        .reset_index(drop=True)
    )

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
    Convert each replicate/sample to log2(CPM+1), then average per compound.
    Depending on target_mode, return either absolute expression log2(CPM+1)
    or L2FC relative to the DMSO control mean.

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

    def _mean_log2cpm(count_matrix: pd.DataFrame) -> pd.Series:
        """CPM-normalize each sample, log-transform, then average replicates."""
        totals = count_matrix.sum(axis=0).replace(0, np.nan)
        cpm = count_matrix.div(totals, axis=1) * 1e6
        return np.log2(cpm + 1).mean(axis=1).fillna(0)

    # Control baseline
    ctrl_log2cpm = _mean_log2cpm(counts[ctrl_ids])

    # Per-compound pseudo-bulk target
    rows = {}
    for compound_id, grp in trt_meta.groupby("compound"):
        cell_cols = [c for c in grp["sequenced_id"].tolist() if c in counts.columns]
        if not cell_cols:
            continue
        cmp_log2cpm = _mean_log2cpm(counts[cell_cols])
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


def contest_gene_filter(target_df: pd.DataFrame) -> list[str]:
    """Return scored contest genes present in the target matrix."""
    contest_genes = load_gene_filter()
    genes = [gene for gene in contest_genes if gene in target_df.columns]
    if not genes:
        raise ValueError("None of the contest gene_filter genes are present in the counts matrix.")
    missing = len(contest_genes) - len(genes)
    if missing:
        print(f"Warning: {missing} contest genes are missing from the counts matrix.")
    return genes


def prediction_frames(
    compounds: list[str],
    user_compound_ids: pd.Series,
    genes: list[str],
    truth_values: np.ndarray,
    pred_values: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build long truth/prediction frames expected by the contest scorer."""
    compound_ids = user_compound_ids.loc[compounds].astype(str).tolist()
    row_index = pd.MultiIndex.from_product(
        [compound_ids, genes],
        names=["compound", "gene_id"],
    )
    truth = pd.DataFrame(
        {
            "expression": truth_values.reshape(-1),
        },
        index=row_index,
    ).reset_index()
    pred = pd.DataFrame(
        {
            "predicted_expression": np.maximum(pred_values.reshape(-1), 0),
        },
        index=row_index,
    ).reset_index()
    return truth, pred


def score_validation_predictions(
    truth: pd.DataFrame,
    pred: pd.DataFrame,
    gene_filter: list[str],
) -> tuple[dict, pd.DataFrame]:
    """Score validation predictions with the contest wMSE metric."""
    weights_source = "official"
    try:
        weights = load_weights_matrix()
        per_compound = score_compounds(
            truth,
            pred,
            gene_filter=gene_filter,
            weights=weights,
        )
    except Exception as exc:
        weights_source = "truth_derived"
        print(
            "Warning: official weights could not be loaded; "
            f"using scorer-derived validation weights instead. Reason: {exc}"
        )
        per_compound = score_compounds(
            truth,
            pred,
            gene_filter=gene_filter,
        )

    board = aggregate_leaderboards(per_compound)
    metrics = {
        "wmse_mean": float(board["wmse_mean"]),
        "wmse_weights_source": weights_source,
    }
    if "wmse_std" in board:
        metrics["wmse_std"] = float(board["wmse_std"])
    return metrics, per_compound


# ---------------------------------------------------------------------------
# PyTorch dataset
# ---------------------------------------------------------------------------

class PerturbationDataset(Dataset):
    def __init__(self, X: np.ndarray, chembl_targets: np.ndarray | None, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.chembl_targets = (
            torch.tensor(chembl_targets, dtype=torch.float32)
            if chembl_targets is not None
            else None
        )
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.chembl_targets is None:
            return self.X[idx], self.y[idx]
        return self.X[idx], self.chembl_targets[idx], self.y[idx]


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


def predict_array(model: nn.Module, X: np.ndarray, chembl_targets: np.ndarray | None = None) -> np.ndarray:
    """Run model inference for a feature matrix."""
    model.eval()
    with torch.no_grad():
        target_tensor = (
            torch.tensor(chembl_targets, dtype=torch.float32)
            if chembl_targets is not None
            else None
        )
        return model(torch.tensor(X, dtype=torch.float32), target_tensor).numpy()


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


class CompoundEncoder(nn.Module):
    """
    Proposed first-stage compound encoder.

    Molecular vector:
    Morgan(2048) + ChemBERTa(384) + UniMol(512) + physicochemical(9)
    -> Linear(2953, 1024) + LayerNorm + GELU + Dropout(0.1).

    Optional ChEMBL targets:
    Linear(n_targets, 64), multiplied by a learned sigmoid confidence gate.
    """

    def __init__(
        self,
        molecular_dim: int = MOLECULAR_INPUT_DIM,
        embed_dim: int = COMPOUND_EMBED_DIM,
        chembl_target_dim: int = 0,
        target_branch_dim: int = TARGET_BRANCH_DIM,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.chembl_target_dim = chembl_target_dim
        self.molecular_encoder = nn.Sequential(
            nn.Linear(molecular_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if chembl_target_dim > 0:
            self.target_encoder = nn.Linear(chembl_target_dim, target_branch_dim)
            self.confidence_gate = nn.Parameter(torch.tensor(0.0))
            self.output_dim = embed_dim + target_branch_dim
        else:
            self.target_encoder = None
            self.confidence_gate = None
            self.output_dim = embed_dim

    def forward(self, molecular_x, chembl_targets=None):
        compound_embedding = self.molecular_encoder(molecular_x)
        if self.target_encoder is None:
            return compound_embedding
        if chembl_targets is None:
            chembl_targets = torch.zeros(
                molecular_x.shape[0],
                self.chembl_target_dim,
                dtype=molecular_x.dtype,
                device=molecular_x.device,
            )
        target_embedding = self.target_encoder(chembl_targets)
        target_embedding = torch.sigmoid(self.confidence_gate) * target_embedding
        return torch.cat([compound_embedding, target_embedding], dim=1)


class PerturbationMLP(nn.Module):
    """
    Maps compound molecular features → predicted per-gene response target.

    Architecture: input projection → residual MLP blocks → output projection.
    Hidden dims are fully configurable; residual blocks are inserted wherever
    consecutive hidden dims are equal.
    """

    def __init__(
        self,
        output_dim: int,
        input_dim: int = MOLECULAR_INPUT_DIM,
        chembl_target_dim: int = 0,
        hidden_dims: list[int] = HIDDEN_DIMS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.compound_encoder = CompoundEncoder(
            molecular_dim=input_dim,
            embed_dim=COMPOUND_EMBED_DIM,
            chembl_target_dim=chembl_target_dim,
            target_branch_dim=TARGET_BRANCH_DIM,
            dropout=dropout,
        )
        layers = []

        prev_dim = self.compound_encoder.output_dim

        # Hidden → hidden (residual where dims match)
        for hidden_dim in hidden_dims:
            if prev_dim == hidden_dim:
                layers.append(ResidualBlock(prev_dim, dropout))
            else:
                layers += [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            prev_dim = hidden_dim

        # Final hidden → output (no activation — regression output)
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, molecular_x, chembl_targets=None):
        compound_embedding = self.compound_encoder(molecular_x, chembl_targets)
        return self.net(compound_embedding)


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

    # Align compound features with target rows
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

    if target_mode == "log2cpm":
        print("Using contest gene_filter genes for training and validation...")
        target_genes = contest_gene_filter(target_df)
    else:
        print("Selecting top HVGs from training compounds...")
        target_genes = select_top_genes(target_df.loc[train_compounds], N_TOP_GENES)
    target_df = target_df[target_genes]

    train_chem = compounds_df.loc[train_compounds].copy()
    val_chem = compounds_df.loc[val_compounds].copy()
    train_chem["compound"] = train_chem.index.astype(str)
    val_chem["compound"] = val_chem.index.astype(str)
    X_train, T_train = build_compound_inputs(train_chem, fit=True)
    y_train = target_df.loc[train_compounds].values.astype(np.float32)
    X_val, T_val = build_compound_inputs(val_chem, fit=False)
    y_val = target_df.loc[val_compounds].values.astype(np.float32)

    dataset = PerturbationDataset(X_train, T_train, y_train)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    chembl_target_dim = 0 if T_train is None else T_train.shape[1]
    model = PerturbationMLP(
        input_dim=X_train.shape[1],
        chembl_target_dim=chembl_target_dim,
        output_dim=y_train.shape[1],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(
        f"\nTraining on {len(dataset)} compounds, validating on {len(val_compounds)}, "
        f"predicting {y_train.shape[1]} genes"
    )
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch in loader:
            if chembl_target_dim > 0:
                xb, tb, yb = batch
            else:
                xb, yb = batch
                tb = None
            optimizer.zero_grad()
            loss = criterion(model(xb, tb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(xb)
        avg = total_loss / len(dataset)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{epochs}  MSE={avg:.4f}")

    val_pred = predict_array(model, X_val, T_val)
    val_metrics = evaluate_arrays(y_val, val_pred)
    if target_mode == "log2cpm":
        if "user_compound_id" in compounds_df.columns:
            user_compound_ids = compounds_df["user_compound_id"].astype("string")
            user_compound_ids = user_compound_ids.where(
                user_compound_ids.notna(),
                pd.Series(compounds_df.index.astype(str), index=compounds_df.index),
            ).astype(str)
        else:
            user_compound_ids = pd.Series(compounds_df.index.astype(str), index=compounds_df.index)
        val_truth, val_prediction_frame = prediction_frames(
            val_compounds,
            user_compound_ids,
            target_genes,
            y_val,
            val_pred,
        )
        wmse_metrics, per_compound = score_validation_predictions(
            val_truth,
            val_prediction_frame,
            target_genes,
        )
        val_metrics.update(wmse_metrics)
    else:
        val_truth = None
        val_prediction_frame = None
        per_compound = None
    print(
        "Validation: "
        f"RMSE={val_metrics['rmse']:.4f}  "
        f"MAE={val_metrics['mae']:.4f}  "
        f"Pearson={val_metrics['pearson']:.4f}"
    )
    if "wmse_mean" in val_metrics:
        print(
            "Contest validation: "
            f"wMSE={val_metrics['wmse_mean']:.4f}  "
            f"weights={val_metrics['wmse_weights_source']}"
        )

    checkpoint_path = run_dir / "model_checkpoint.pt"
    feature_state_path = run_dir / "feature_state.pkl"
    gene_list_path = run_dir / "top_genes.txt"
    metrics_path = run_dir / "validation_metrics.json"
    metadata_path = run_dir / "run_metadata.json"

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": X_train.shape[1],
            "chembl_target_dim": chembl_target_dim,
            "output_dim": y_train.shape[1],
        },
        checkpoint_path,
    )
    with open(feature_state_path, "wb") as f:
        pickle.dump(_get_feature_state(), f)
    with open(gene_list_path, "w") as f:
        f.write("\n".join(target_genes))
    with open(metrics_path, "w") as f:
        json.dump(val_metrics, f, indent=2)
    if val_truth is not None and val_prediction_frame is not None and per_compound is not None:
        val_truth.to_csv(run_dir / "validation_truth.csv", index=False)
        val_prediction_frame.to_csv(run_dir / "validation_predictions.csv", index=False)
        per_compound.to_csv(run_dir / "validation_per_compound.csv", index=False)
    with open(metadata_path, "w") as f:
        json.dump(
            {
                "target_mode": target_mode,
                "feature_mode": FEATURE_MODE,
                "molecular_input_dim": int(X_train.shape[1]),
                "chembl_target_dim": int(chembl_target_dim),
                "n_target_genes": len(target_genes),
                "target_gene_source": "gene_filter" if target_mode == "log2cpm" else "top_hvg",
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
    model = PerturbationMLP(
        input_dim=ckpt["input_dim"],
        chembl_target_dim=ckpt.get("chembl_target_dim", 0),
        output_dim=ckpt["output_dim"],
    )
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
        matched_features = compounds_by_uuid.loc[test_ids].copy()
        matched_features["compound"] = test_ids.values
        return matched_features.reset_index(drop=True)

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
            matched_features["compound"] = test_ids.values
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
    test_df["compound"] = test_ids.values
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
    X, T = build_compound_inputs(test_chem, fit=False)

    preds = predict_array(model, X, T)

    target_col = "predicted_lfc" if target_mode == "l2fc" else "predicted_log2cpm"

    records = [
        (cmpd, gene, float(preds[i, j]))
        for i, cmpd in enumerate(test_df["compound"].tolist())
        for j, gene in enumerate(top_genes)
    ]
    out = pd.DataFrame(records, columns=["compound", "gene_id", "prediction"])
    out[target_col] = out["prediction"]
    if target_mode == "log2cpm":
        out["predicted_expression"] = np.maximum(out["prediction"], 0)

    out_path = run_dir / "predictions.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path}  ({len(out):,} rows)")
    return out

    
def verify(run_dir: Path | None = None) -> pd.DataFrame:
    """Verify model predictions against the held-out validation set."""
    model, top_genes, run_dir, metadata = load_trained_model(run_dir)
    target_mode = metadata.get("target_mode", "l2fc")
    val_compounds = metadata.get("val_compounds")

    if not val_compounds:
        raise ValueError(
            "No validation split found in run metadata. Train with --val-size to save a validation set."
        )

    counts, meta, compounds_df = load_raw_data()
    target_df, _ = compute_pseudobulk_targets(counts, meta, target_mode=target_mode)
    target_df = target_df.loc[val_compounds, top_genes]

    compounds_df = compounds_df.set_index("compound")
    if "user_compound_id" in compounds_df.columns:
        user_compound_ids = compounds_df["user_compound_id"].astype("string")
        user_compound_ids = user_compound_ids.where(
            user_compound_ids.notna(),
            pd.Series(compounds_df.index.astype(str), index=compounds_df.index),
        ).astype(str)
    else:
        user_compound_ids = pd.Series(compounds_df.index.astype(str), index=compounds_df.index)

    val_features = compounds_df.loc[val_compounds].copy()
    val_features["compound"] = val_features.index.astype(str)
    X_val, T_val = build_compound_inputs(val_features, fit=False)

    preds = predict_array(model, X_val, T_val)
    truth, pred = prediction_frames(
        val_compounds,
        user_compound_ids,
        top_genes,
        target_df.to_numpy(dtype=np.float32),
        preds,
    )

    truth.to_csv(run_dir / "validation_truth.csv", index=False)
    pred.to_csv(run_dir / "validation_predictions.csv", index=False)
    metrics = evaluate_arrays(target_df.to_numpy(dtype=np.float32), preds)
    if target_mode == "log2cpm":
        wmse_metrics, per_compound = score_validation_predictions(truth, pred, top_genes)
        metrics.update(wmse_metrics)
        per_compound.to_csv(run_dir / "validation_per_compound.csv", index=False)
    with open(run_dir / "validation_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Wrote validation files to {run_dir}/")
    print(
        f"Verification metrics: RMSE={metrics['rmse']:.4f}, "
        f"MAE={metrics['mae']:.4f}, Pearson={metrics['pearson']:.4f}"
    )
    if "wmse_mean" in metrics:
        print(
            f"Contest validation: wMSE={metrics['wmse_mean']:.4f}, "
            f"weights={metrics['wmse_weights_source']}"
        )
    return pred


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

    for metric in ("wmse_mean", "rmse", "mae", "pearson"):
        old_value = old_metrics.get(metric)
        new_value = new_metrics.get(metric)
        if old_value is None or new_value is None:
            continue
        if metric in ("wmse_mean", "rmse", "mae"):
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
    print("  wMSE/RMSE/MAE improve when they decrease.")
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
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the saved model against the held-out validation split",
    )
    args = parser.parse_args()

    if not args.train and not args.predict and not args.compare_runs and not args.verify:
        parser.print_help()

    if args.compare_runs:
        comparison = compare_runs(Path(args.compare_runs[0]), Path(args.compare_runs[1]))
        print_run_comparison(comparison)

    run_dir = Path(args.run_dir) if args.run_dir else None

    if args.train:
        run_dir = train(target_mode=args.target_mode, val_size=args.val_size)
    if args.predict:
        predict(args.predict, run_dir=run_dir)
    if args.verify:
        verify(run_dir=run_dir)
