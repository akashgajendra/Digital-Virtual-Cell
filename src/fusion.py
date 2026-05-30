"""
Layer 3 — Fusion Model + Gene Program Bottleneck.

Fuses compound_embedding (from Layer 1) + gated auxiliary targets (64)
+ cell_state (from Layer 2) → shared representation → gene program bottleneck
→ predicted_expression (12,995 genes) + predicted_variance.
"""

import numpy as np
import torch
import torch.nn as nn

from src.config import (
    DROPOUT,
    N_AUX_TARGET,
    N_CELL_STATE,
    N_COMPOUND_EMB,
    N_OUTPUT_GENES,
    N_PROGRAMS,
)


# ---------------------------------------------------------------------------
# Layer 3: Fusion Model + Gene Program Bottleneck
# ---------------------------------------------------------------------------

class FusionModel(nn.Module):
    """
    Fuses compound embedding (1024) + gated auxiliary ChEMBL targets (64)
    + cell state (117) → shared representation → gene program bottleneck.

    Outputs:
        predicted_expression : (batch, 12995)  mean, clamped ≥ 0
        predicted_variance   : (batch, 12995)  uncertainty (Softplus, always > 0)
        gene_programs        : (batch, 128)    activations for Layer 4 interpretation
    """

    def __init__(self, n_genes: int = N_OUTPUT_GENES, n_programs: int = N_PROGRAMS):
        super().__init__()
        self.n_genes    = n_genes
        self.n_programs = n_programs
        in_dim          = N_COMPOUND_EMB + N_AUX_TARGET + N_CELL_STATE  # 1205

        # Learned gate on the auxiliary branch — closes to zero when targets are missing
        self.aux_gate = nn.Parameter(torch.zeros(1))

        # Fusion MLP: 1205 → 512 → 256
        self.fusion = nn.Sequential(
            nn.Linear(in_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(512, 256),    nn.LayerNorm(256), nn.GELU(),
        )

        # Mean head: 256 → 128 gene programs → 12,995 genes
        self.program_encoder = nn.Linear(256, n_programs)
        self.gene_decoder    = nn.Linear(n_programs, n_genes, bias=False)  # NMF-initialised

        # Variance head: parallel path, always positive
        self.var_head = nn.Sequential(
            nn.Linear(256, n_programs), nn.Softplus(),
            nn.Linear(n_programs, n_genes), nn.Softplus(),
        )

    def init_with_nmf(self, H: np.ndarray):
        """
        Initialise gene_decoder weights from NMF gene loadings.
        H shape must be (n_programs, n_genes).
        Each row of H is one biological program; the decoder learns to
        weight them when reconstructing expression.
        """
        assert H.shape == (self.n_programs, self.n_genes), \
            f"NMF H shape {H.shape} ≠ expected ({self.n_programs}, {self.n_genes})"
        with torch.no_grad():
            self.gene_decoder.weight.copy_(torch.tensor(H, dtype=torch.float32))
        print(f"Gene decoder initialised from NMF "
              f"({H.shape[0]} programs × {H.shape[1]} genes)")

    def forward(
        self,
        compound_embedding: torch.Tensor,   # (batch, 1024)
        aux_target:         torch.Tensor,   # (batch, 64)   zeros when unavailable
        cell_state:         torch.Tensor,   # (batch, 117)
        dmso_baseline:      torch.Tensor,   # (12995,) or (batch, 12995)
    ):
        gate   = torch.sigmoid(self.aux_gate)
        x      = torch.cat([compound_embedding, gate * aux_target, cell_state], dim=-1)
        shared = self.fusion(x)                             # (batch, 256)

        programs             = self.program_encoder(shared) # (batch, 128)
        delta                = self.gene_decoder(programs)  # (batch, 12995)
        predicted_expression = torch.clamp(delta + dmso_baseline, min=0.0)
        predicted_variance   = self.var_head(shared)        # (batch, 12995)

        return predicted_expression, predicted_variance, programs
