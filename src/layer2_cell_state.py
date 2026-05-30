"""
Layer 2 — CellStateEncoder.

Produces a 117-dim cell context vector per compound:
    [DMSO PCA 50-dim | batch one-hot 3-dim | LINCS KNN 64-dim]

Call fit() once on training data.  LINCS features are optional;
call load_lincs() to enable them (requires rdkit: uv add rdkit).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

from src.config import MAX_BATCHES
from src.data import quality_filter


class CellStateEncoder:

    def __init__(self, n_pca: int = 50, n_batches: int = MAX_BATCHES):
        self.n_pca     = n_pca
        self.n_batches = n_batches

        self.pca:               PCA | None        = None
        self.dmso_pca_baseline: np.ndarray | None = None   # (50,)

        # LINCS (optional — populated by load_lincs())
        self._lincs_profiles:     np.ndarray | None = None
        self._lincs_fingerprints: np.ndarray | None = None
        self._lincs_linear:       nn.Linear | None  = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, counts: pd.DataFrame, meta: pd.DataFrame):
        """Fit PCA on clean DMSO cells."""
        clean    = quality_filter(meta)
        ctrl_ids = clean.loc[clean["is_control"].astype(bool), "sequenced_id"].astype(str).tolist()

        counts_idx         = counts.set_index("gene_id")
        counts_idx.columns = counts_idx.columns.astype(str)
        ctrl_ids           = [c for c in ctrl_ids if c in counts_idx.columns]

        ctrl_mat = counts_idx[ctrl_ids].T.values.astype(np.float32)  # (n_cells, n_genes)
        totals   = ctrl_mat.sum(axis=1, keepdims=True)
        totals   = np.where(totals == 0, 1, totals)
        log2cpm  = np.log2(ctrl_mat / totals * 1e6 + 1)

        self.pca           = PCA(n_components=self.n_pca)
        pca_out            = self.pca.fit_transform(log2cpm)   # (n_cells, 50)
        self.dmso_pca_baseline = pca_out.mean(axis=0)          # (50,)
        print(f"CellStateEncoder: PCA on {len(ctrl_ids)} DMSO cells "
              f"(variance explained: {self.pca.explained_variance_ratio_.sum():.1%})")

    # ------------------------------------------------------------------
    # Optional: LINCS KNN features
    # ------------------------------------------------------------------

    def load_lincs(self, profiles_path: str, smiles_path: str):
        """
        Load LINCS L1000 profiles to enable KNN compound features.
        profiles_path: parquet/csv with columns [compound_id, <978 gene cols>]
        smiles_path:   csv with columns [compound_id, smiles]
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
        except ImportError:
            print("rdkit not installed — LINCS KNN disabled. Run: uv add rdkit")
            return

        from pathlib import Path
        ext      = Path(profiles_path).suffix
        profiles = pd.read_parquet(profiles_path) if ext == ".parquet" else pd.read_csv(profiles_path)
        smi_df   = pd.read_csv(smiles_path)

        gene_cols = [c for c in profiles.columns if c != "compound_id"]
        self._lincs_profiles = profiles[gene_cols].values.astype(np.float32)

        fps = []
        for smi in smi_df["smiles"]:
            mol = Chem.MolFromSmiles(str(smi)) if pd.notna(smi) else None
            fp  = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048) if mol else None
            fps.append(np.array(fp) if fp is not None else np.zeros(2048, dtype=np.float32))
        self._lincs_fingerprints = np.stack(fps).astype(np.float32)
        self._lincs_linear       = nn.Linear(5 * len(gene_cols), 64)
        print(f"LINCS loaded: {len(profiles)} compounds, {len(gene_cols)} genes")

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def _knn_feature(self, smiles: str | None) -> torch.Tensor:
        """64-dim LINCS KNN feature for one compound. Returns zeros if unavailable."""
        if self._lincs_profiles is None or smiles is None:
            return torch.zeros(64)
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return torch.zeros(64)
            query = np.array(
                AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048),
                dtype=np.float32,
            )
            inter = (self._lincs_fingerprints * query).sum(axis=1)
            union = np.clip(self._lincs_fingerprints + query, 0, 1).sum(axis=1)
            top5  = np.argpartition(inter / (union + 1e-8), -5)[-5:]
            raw   = torch.tensor(self._lincs_profiles[top5].flatten())
            with torch.no_grad():
                return self._lincs_linear(raw)
        except Exception:
            return torch.zeros(64)

    def encode(self, smiles: str | None = None, batch_id: int = 0) -> torch.Tensor:
        """Return 117-dim cell state tensor for one compound."""
        assert self.pca is not None, "Call fit() before encode()."
        pca_part   = torch.tensor(self.dmso_pca_baseline, dtype=torch.float32)   # (50,)
        batch_part = F.one_hot(torch.tensor(batch_id), self.n_batches).float()   # (3,)
        lincs_part = self._knn_feature(smiles)                                   # (64,)
        return torch.cat([pca_part, batch_part, lincs_part])                     # (117,)

    def encode_batch(self, smiles_list: list, batch_id: int = 0) -> torch.Tensor:
        """Encode a list of SMILES strings. Returns (n, 117)."""
        return torch.stack([self.encode(s, batch_id) for s in smiles_list])
