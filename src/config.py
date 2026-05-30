"""
Central configuration — change hyperparameters here.
Everything else imports from this file; nothing here imports from src/.
"""

from pathlib import Path
import vcpi_prediction_contest as vcpi

# Paths
DATASETS_DIR = Path("datasets")
RUNS_DIR     = Path("runs")

# Compound features — purity_pct excluded (all-null in this dataset)
COMPOUND_FEATURE_COLS = [
    "molecular_weight", "log_p", "tpsa",
    "num_rotatable_bonds", "num_h_acceptors", "num_h_donors",
    "num_atoms", "num_bonds",
]

# Gene space — fixed by the contest scoring package, do not change
OUTPUT_GENES:  list[str] = vcpi.load_gene_filter()   # 12,995 ENSEMBL IDs
N_OUTPUT_GENES: int      = len(OUTPUT_GENES)          # 12,995

# Architecture dims — coordinate N_COMPOUND_EMB with the Layer 1 team
N_COMPOUND_EMB = 1024   # Layer 1 output dim
N_CELL_STATE   = 117    # 50 (DMSO PCA) + 3 (batch one-hot) + 64 (LINCS KNN)
N_AUX_TARGET   = 64     # ChEMBL gated auxiliary branch
N_PROGRAMS     = 128    # gene program bottleneck width
MAX_BATCHES    = 3      # batch one-hot size: 1 now, room for Tahoe/GDPx

# Training
BATCH_SIZE = 64
EPOCHS     = 100
LR         = 3e-4
DROPOUT    = 0.1
