"""
Data loading, QC filtering, normalization, NMF, and PyTorch dataset.
Uses the official vcpi_prediction_contest normalizer throughout.
"""

import glob

import numpy as np
import pandas as pd
import torch
import vcpi_prediction_contest as vcpi
from sklearn.decomposition import NMF
from torch.utils.data import Dataset

from src.config import (
    DATASETS_DIR,
    N_AUX_TARGET,
    N_PROGRAMS,
    OUTPUT_GENES,
)


# ---------------------------------------------------------------------------
# Raw data loading
# ---------------------------------------------------------------------------

def _discover_file(pattern: str):
    matches = sorted(glob.glob(str(DATASETS_DIR / pattern)))
    if not matches:
        raise FileNotFoundError(f"No file matching '{pattern}' in {DATASETS_DIR}.")
    from pathlib import Path
    return Path(matches[0])


def load_raw_data():
    """Load counts, metadata, and compound features from datasets/."""
    counts_path    = _discover_file("vcpi_*_counts.parquet")
    meta_path      = _discover_file("metadata-*.csv")
    compounds_path = _discover_file("compounds-*.csv")
    print(f"Counts   : {counts_path}")
    print(f"Metadata : {meta_path}")
    print(f"Compounds: {compounds_path}")

    counts       = pd.read_parquet(counts_path)
    meta         = pd.read_csv(meta_path)
    compounds_df = pd.read_csv(compounds_path)
    print(f"Counts shape: {counts.shape[0]} genes × {counts.shape[1]-1} cells")
    return counts, meta, compounds_df


# ---------------------------------------------------------------------------
# Quality control
# ---------------------------------------------------------------------------

def quality_filter(meta: pd.DataFrame) -> pd.DataFrame:
    """Remove edge-well cells and those with >20% mitochondrial reads."""
    before = len(meta)
    clean  = meta[
        (~meta["is_edge"].astype(bool)) &
        (meta["percent_mitochondrial"] < 20)
    ].copy()
    print(f"QC: kept {len(clean)}/{before} cells "
          f"(removed {before - len(clean)} edge/high-mito)")
    return clean


# ---------------------------------------------------------------------------
# Expression normalization
# ---------------------------------------------------------------------------

def build_expression_table(counts: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """
    Official normalizer: log2(CPM+1) averaged per compound, QC-filtered.
    Returns long-format (compound, gene_id, expression) for the 12,995 scored genes.
    """
    clean = quality_filter(meta)
    expr  = vcpi.counts_to_expression(counts, clean)
    return expr[expr[vcpi.GENE_COL].isin(OUTPUT_GENES)]


def compute_dmso_baseline(counts: pd.DataFrame, meta: pd.DataFrame) -> pd.Series:
    """
    Mean log2(CPM+1) across clean DMSO cells for the 12,995 scored genes.
    Added back in Layer 3 so the model predicts deltas from baseline.
    Returns a Series indexed by gene_id, shape (12,995,).
    """
    clean    = quality_filter(meta)
    ctrl_ids = clean.loc[clean["is_control"].astype(bool), "sequenced_id"].astype(str).tolist()

    counts_idx         = counts.set_index("gene_id")
    counts_idx.columns = counts_idx.columns.astype(str)
    ctrl_counts        = counts_idx[[c for c in ctrl_ids if c in counts_idx.columns]]

    lib  = ctrl_counts.sum(axis=0).replace(0, np.nan)
    cpm  = ctrl_counts.div(lib, axis=1) * 1e6
    log2 = np.log2(cpm + 1.0)
    return log2.mean(axis=1).reindex(OUTPUT_GENES).fillna(0).astype(np.float32)


def compute_gene_weights(counts: pd.DataFrame, meta: pd.DataFrame) -> pd.Series:
    """
    Mejia weights — the exact per-gene weights used by the scoring metric.
    Returns a Series indexed by gene_id.
    """
    clean   = quality_filter(meta)
    weights = vcpi.compute_mejia_weights(counts, clean)
    return weights.reindex(OUTPUT_GENES).fillna(weights.mean())


# ---------------------------------------------------------------------------
# NMF: initialises the gene decoder in Layer 3
# ---------------------------------------------------------------------------

def compute_nmf_factors(
    expr_wide:    pd.DataFrame,
    n_components: int = N_PROGRAMS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Decompose pseudo-bulk expression into gene programs: X ≈ W × H.
      W  (n_compounds × n_programs) — program activation per compound
      H  (n_programs  × n_genes)    — gene loadings per program → init gene decoder
    """
    print(f"NMF: {n_components} programs on "
          f"{expr_wide.shape[0]} compounds × {expr_wide.shape[1]} genes ...")
    X   = np.clip(expr_wide.values.astype(np.float32), 0, None)
    nmf = NMF(n_components=n_components, init="nndsvda", max_iter=500, random_state=42)
    W   = nmf.fit_transform(X)
    H   = nmf.components_
    print(f"NMF reconstruction error: {nmf.reconstruction_err_:.2f}")
    return W, H


# ---------------------------------------------------------------------------
# PyTorch dataset
# ---------------------------------------------------------------------------

class PerturbationDataset(Dataset):
    """One sample = one compound: physio features + cell state → target expression."""

    def __init__(
        self,
        compound_features: np.ndarray,            # (n, 8)     StandardScaler-normalised
        cell_states:       np.ndarray,            # (n, 117)
        expression:        np.ndarray,            # (n, 12995) training targets
        dmso_baseline:     np.ndarray,            # (12995,)   broadcast during training
        aux_targets:       np.ndarray | None = None,  # (n, 64) ChEMBL targets; zeros if None
    ):
        self.X_comp = torch.tensor(compound_features, dtype=torch.float32)
        self.X_cell = torch.tensor(cell_states,       dtype=torch.float32)
        self.y      = torch.tensor(expression,        dtype=torch.float32)
        self.dmso   = torch.tensor(dmso_baseline,     dtype=torch.float32)
        self.aux    = (
            torch.tensor(aux_targets, dtype=torch.float32)
            if aux_targets is not None
            else torch.zeros(len(compound_features), N_AUX_TARGET)
        )

    def __len__(self):
        return len(self.X_comp)

    def __getitem__(self, idx):
        return self.X_comp[idx], self.aux[idx], self.X_cell[idx], self.dmso, self.y[idx]
