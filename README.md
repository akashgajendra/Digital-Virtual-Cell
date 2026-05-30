# Digital Virtual Cell — Drug-seq Perturbation Model

Predicts per-gene perturbation responses for unseen compounds using pseudo-bulk
aggregates from VCPI Drug-seq data and a molecular encoder trained from SMILES.
The default target is now absolute pseudo-bulk expression, `log2(CPM+1)`. The
old DMSO-relative L2FC target is still available with `--target-mode l2fc`.

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

Creates a timestamped run folder, e.g. `runs/20260530_143022/`, containing:
- `model_checkpoint.pt`
- `feature_state.pkl`
- `top_genes.txt`
- `validation_metrics.json`
- `run_metadata.json`

**Train with the old L2FC target:**
```bash
uv run python model.py --train --target-mode l2fc
```

**Create the official test CSV from the contest package:**
```bash
uv run python generate_predictions.py
```

**Predict:**
```bash
uv run python model.py --predict datasets/test_compounds.csv
```

Loads the latest run automatically and writes `runs/<timestamp>/predictions.csv`.

**Train + predict in one shot:**
```bash
uv run python model.py --train --predict datasets/test_compounds.csv
```

**Load a specific run (not the latest):**
```bash
uv run python model.py --predict datasets/test_compounds.csv --run-dir runs/20260530_143022
```

## Comparing old vs new models

`test_compounds.csv` has no ground-truth expression values, so predictions on
that file alone cannot prove improvement. To compare models, train both versions
with the same validation split and compare their saved validation metrics:

```bash
uv run python model.py --train --target-mode l2fc
uv run python model.py --train --target-mode log2cpm
uv run python model.py --compare-runs runs/OLD_L2FC_RUN runs/NEW_LOG2CPM_RUN
```

Improvement criteria:
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
| `f1361a6b-...` | `ENSG00000290825` | `0.42` |

New `log2cpm` runs write `prediction` and `predicted_log2cpm`; old `l2fc` runs
write `prediction` and `predicted_lfc`.

## Project structure

```
model.py          # model definition, training, prediction pipeline
pyproject.toml    # dependencies (managed by uv)
uv.lock           # locked dependency versions
requirements.txt  # pip-compatible dependency list
datasets/         # gitignored — add your VCPI files here
runs/             # gitignored — created automatically on each training run
```
