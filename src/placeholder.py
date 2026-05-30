"""
Layer 1 — Compound Representation (placeholder).

TEAMMATES: this is your file. Replace PlaceholderLayer1 with your real encoder.

Interface contract:
  Input  shape : (batch, 8)    StandardScaler-normalised physicochemical features
  Output shape : (batch, 1024) compound_embedding fed into Layer 3 FusionModel
  Output dim   : must equal N_COMPOUND_EMB = 1024

Real Layer 1 architecture (to be implemented):
  SMILES → Morgan Fingerprints (2048)
         → ChemBERTa (384)
         → UniMol (512)
         → Physicochemical (9)
  Concatenate → 2953 dims
  Linear(2953 → 1024) + LayerNorm + GELU + Dropout(0.1)
  → compound_embedding: 1024 dims
"""

import torch
import torch.nn as nn

from src.config import COMPOUND_FEATURE_COLS, DROPOUT, N_COMPOUND_EMB


class PlaceholderLayer1(nn.Module):
    """Temporary: maps physicochemical features (8 dims) → 1024-dim compound_embedding."""

    def __init__(self, input_dim: int = len(COMPOUND_FEATURE_COLS)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(256, 512),       nn.LayerNorm(512), nn.GELU(),
            nn.Linear(512, N_COMPOUND_EMB), nn.LayerNorm(N_COMPOUND_EMB),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
