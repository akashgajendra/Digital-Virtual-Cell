# Digital Virtual Cell — Drug-seq Perturbation Model

Predicts per-gene log2 fold-change (L2FC) for unseen compounds using pseudo-bulk
aggregates from VCPI Drug-seq data and a residual MLP trained on compound
physicochemical features.

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

## Running the model

**Train:**
```bash
uv run python model.py --train
```

Creates a timestamped run folder, e.g. `runs/20260530_143022/`, containing:
- `model_checkpoint.pt`
- `feature_scaler.pkl`
- `top_genes.txt`

**Predict:**
```bash
uv run python model.py --predict datasets/test_compounds.csv
```

Loads the latest run automatically and writes `runs/<timestamp>/predictions.parquet`.

**Train + predict in one shot:**
```bash
uv run python model.py --train --predict datasets/test_compounds.csv
```

**Load a specific run (not the latest):**
```bash
uv run python model.py --predict datasets/test_compounds.csv --run-dir runs/20260530_143022
```

## Output format

`predictions.parquet` has one row per `(compound, gene_id)` pair:

| compound | gene_id | predicted_lfc |
|---|---|---|
| `f1361a6b-...` | `ENSG00000290825` | `0.42` |

## Project structure

```
model.py          # model definition, training, prediction pipeline
pyproject.toml    # dependencies (managed by uv)
uv.lock           # locked dependency versions
requirements.txt  # pip-compatible dependency list
datasets/         # gitignored — add your VCPI files here
runs/             # gitignored — created automatically on each training run
```
