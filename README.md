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
| `unimol_embeddings.csv` | Optional precomputed UniMol embeddings (`unimol_0` ... `unimol_511`) |

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

## Architecture

The compound encoder builds the proposed 2953-dimensional molecular input:

| Source | Dimensions |
|---|---:|
| Morgan fingerprint | 2048 |
| ChemBERTa embedding | 384 |
| UniMol embedding | 512 |
| Physicochemical descriptors | 9 |

That vector is projected with `Linear(2953 -> 1024)`, `LayerNorm`, `GELU`, and
`Dropout(0.1)`. If ChEMBL target columns are present in `compounds-*.csv`, they
are passed through a gated auxiliary branch: `Linear(n_targets -> 64)` multiplied
by a learned sigmoid confidence gate and concatenated to the compound embedding.

UniMol embeddings are loaded from `datasets/unimol_embeddings.csv` when present.
If the file is absent, the UniMol slot is filled with zeros so the model remains
runnable with the same input shape.

## Running the model

**Train:**
```bash
uv run python model.py --train
```

**Predict** (test compounds are bundled — no path needed):
```bash
uv run python model.py --predict
```

Loads the latest run automatically and writes `runs/<timestamp>/predictions.csv`.

**Create final submission files:**
```bash
uv run python make_submission.py --run-dir runs/<timestamp>
```

Writes both `runs/<timestamp>/submission.csv` and
`runs/<timestamp>/submission.parquet` with the required columns:
`compound`, `gene_id`, and `predicted_expression`.
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

## Comparing old vs new models

`test_compounds.csv` has no ground-truth expression values, so predictions on
that file alone cannot prove improvement. Training automatically holds out a
validation split, writes contest-shaped truth/prediction CSVs, and reports
weighted MSE (`wMSE`) when `--target-mode log2cpm` is used.

To re-run validation for a saved run:

```bash
uv run python model.py --verify --run-dir runs/20260530_143022
```

To compare models, train both versions with the same validation split and
compare their saved validation metrics:

```bash
uv run python model.py --train --target-mode l2fc
uv run python model.py --train --target-mode log2cpm
uv run python model.py --compare-runs runs/OLD_L2FC_RUN runs/NEW_LOG2CPM_RUN
```

Improvement criteria:
- `wMSE` decreases: contest weighted prediction error is lower.
- `RMSE` decreases: average squared prediction error is lower.
- `MAE` decreases: average absolute prediction error is lower.
- `Pearson` increases: predicted and observed gene-response patterns align more
  strongly.

For `log2cpm` vs `l2fc`, the fairest primary criteria are validation `RMSE` and
`MAE` on the held-out labeled compounds, plus Pearson for pattern agreement.
Predictions on `test_compounds.csv` are useful for submission generation, but
they do not contain observed values and therefore are not an improvement test.
The comparison command prints the old value, new value, delta, percent change,
and whether each criterion improved.

## Output format

`predictions.csv` has one row per `(compound, gene_id)` pair:

| compound | gene_id | prediction |
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

New `log2cpm` runs write `prediction` and `predicted_log2cpm`; old `l2fc` runs
write `prediction` and `predicted_lfc`.

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
