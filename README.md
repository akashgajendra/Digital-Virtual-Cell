# Digital Virtual Cell — Drug-seq Perturbation Model

Predicts per-gene expression for unseen compounds using pseudo-bulk aggregates
from VCPI Drug-seq data. Four-layer architecture: compound representation →
cell state → fusion + gene program bottleneck → interpretation.

## Setup

**Recommended (uv):**
```bash
uv sync
```

**Alternatively (pip):**
```bash
pip install -r requirements.txt
```

## Dataset files

Place these files in the `datasets/` folder (they are gitignored):

| File | Description |
|---|---|
| `vcpi_*_counts.parquet` | Gene expression counts (genes × cells) |
| `metadata-*.csv` | Per-cell metadata (compound, cell line, controls) |
| `compounds-*.csv` | Compound physicochemical features + SMILES |

Files are discovered by glob, so the exact filename (e.g. `tvc-qnu-012`) doesn't matter.

## External data (one-time setup)

### ChEMBL targets, gene symbols, MSigDB

Fetches from free APIs — no account needed:
```bash
uv run python -m src.fetch_external
```

### LINCS L1000 (Layer 2 KNN feature)

**Step 1 — Download these 4 files** from [clue.io/data/CMap2020#LINCS2020](https://clue.io/data/CMap2020#LINCS2020) into `datasets/lincs/`:

| File | Size |
|---|---|
| `level5_beta_trt_cp_n720216x12328.gctx` | ~10 GB |
| `geneinfo_beta.txt` | small |
| `siginfo_beta.txt` | small |
| `compoundinfo_beta.txt` | small |

**Step 2 — Process the metadata** (run as soon as the 3 small files are downloaded, before the big file finishes):
```bash
uv run python -m src.fetch_external --lincs-meta-only
```
Saves `datasets/lincs_smiles.csv` and `datasets/lincs_landmark_genes.txt`.

**Step 3 — Convert the .gctx** (run after the big file finishes downloading):
```bash
uv add cmapPy
uv run python -m src.fetch_external --lincs
```
Saves `datasets/lincs_profiles.parquet`. Training will pick it up automatically.

> **Note:** `datasets/lincs/` is gitignored. Each teammate needs to download the files themselves.

## Running the model

**Train:**
```bash
uv run python model.py --train
```

**Predict** (test compounds are bundled — no path needed):
```bash
uv run python model.py --predict
```

**Train + predict in one shot:**
```bash
uv run python model.py --train --predict
```

**Interpret gene programs** (run after predict):
```bash
uv run python model.py --interpret
```

**Use a specific run instead of the latest:**
```bash
uv run python model.py --predict --run-dir runs/20260530_143022
```

Each run creates a timestamped folder `runs/<timestamp>/` containing:
`model_checkpoint.pt`, `feat_scaler.pkl`, `cell_enc.pkl`, `nmf_H.npy`,
`dmso_mean.npy`, `predictions.parquet`, `uncertainty.npy`,
`program_activations.npy`, `program_activations.csv`

## Output format

`predictions.parquet` has one row per `(compound, gene_id)` pair — 1,064 compounds × 12,995 genes:

| compound | gene_id | predicted_expression |
|---|---|---|
| `9321914` | `ENSG00000000003` | `3.42` |

## Architecture

```
Layer 1  placeholder.py    Compound representation → 1024-dim embedding
         (teammates)       SMILES → Morgan FP + ChemBERTa + UniMol + physio

Layer 2  cell_state.py     Cell state → 117-dim context vector
                           DMSO PCA (50) + batch one-hot (3) + LINCS KNN (64)

Layer 3  fusion.py         Fusion + gene program bottleneck
                           (1024 + 64 + 117) → 512 → 256 → 128 programs → 12,995 genes
                           Parallel variance head for uncertainty estimation
                           Gene decoder initialised from NMF factors

Layer 4  interpret.py      Interpretation only — never modifies predictions
                           Gene programs → GSEA → pathway names
```

## Project structure

```
model.py                    CLI entry point (--train / --predict / --interpret)
src/
  config.py                 all hyperparameters and constants
  data.py                   loading, QC filtering, normalization, NMF, Dataset
  runs.py                   timestamped run folder helpers
  loss.py                   wmse_loss (Mejia weights), kl_variance_loss
  train.py                  training pipeline
  predict.py                prediction pipeline
  placeholder.py            ← TEAMMATES: replace this with your Layer 1 encoder
  cell_state.py             CellStateEncoder (Layer 2)
  fusion.py                 FusionModel (Layer 3)
  interpret.py              interpret_programs, run_gsea_on_programs (Layer 4)
pyproject.toml              dependencies (managed by uv)
uv.lock                     locked dependency versions
requirements.txt            pip-compatible dependency list
datasets/                   gitignored — add your VCPI files here
runs/                       gitignored — one folder created per training run
```
