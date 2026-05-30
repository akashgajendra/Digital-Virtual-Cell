"""Prediction pipeline — loads artifacts and writes predictions.parquet."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import vcpi_prediction_contest as vcpi

from src.config import COMPOUND_FEATURE_COLS, N_AUX_TARGET, OUTPUT_GENES
from src.data import load_raw_data
from src.layer3_fusion import FusionModel, PlaceholderLayer1
from src.runs import latest_run_dir


def load_artifacts(run_dir: Path | None = None):
    """Load model, scaler, cell encoder, and DMSO baseline from a run directory."""
    if run_dir is None:
        run_dir = latest_run_dir()
    print(f"Loading from {run_dir}/")

    ckpt   = torch.load(run_dir / "model_checkpoint.pt", weights_only=True)
    layer1 = PlaceholderLayer1(input_dim=ckpt["input_dim"])
    layer1.load_state_dict(ckpt["layer1_state"]); layer1.eval()

    model = FusionModel(n_genes=ckpt["n_genes"], n_programs=ckpt["n_programs"])
    model.load_state_dict(ckpt["model_state"]); model.eval()

    with open(run_dir / "feat_scaler.pkl", "rb") as f: feat_scaler = pickle.load(f)
    with open(run_dir / "cell_enc.pkl",    "rb") as f: cell_enc    = pickle.load(f)

    dmso_mean = np.load(run_dir / "dmso_mean.npy")
    return layer1, model, feat_scaler, cell_enc, dmso_mean, run_dir


def predict(run_dir: Path | None = None) -> pd.DataFrame:
    """
    Predict for all 1,064 test compounds (bundled in the vcpi package).
    Writes to runs/<timestamp>/predictions.parquet.
    Output columns: (compound, gene_id, predicted_expression).
    """
    layer1, model, feat_scaler, cell_enc, dmso_mean, run_dir = load_artifacts(run_dir)

    # Test compounds are bundled in the vcpi scoring package
    test_df = vcpi.load_test_compounds()
    print(f"Test compounds: {len(test_df)}")

    # test_df['compound'] is user_compound_id (numeric string)
    # Join with training compounds CSV to get physicochemical features
    _, _, compounds_df = load_raw_data()
    compounds_df["user_compound_id"] = compounds_df["user_compound_id"].astype(str)
    joined = test_df.merge(
        compounds_df[["user_compound_id"] + COMPOUND_FEATURE_COLS],
        left_on="compound", right_on="user_compound_id", how="left",
    )
    missing = joined[COMPOUND_FEATURE_COLS[0]].isna().sum()
    if missing:
        print(f"Warning: {missing} test compounds missing features — using zeros.")
    joined[COMPOUND_FEATURE_COLS] = joined[COMPOUND_FEATURE_COLS].fillna(0)

    X_raw       = joined[COMPOUND_FEATURE_COLS].values.astype("float32")
    X           = feat_scaler.transform(X_raw)
    smiles_list = joined["smiles"].tolist() if "smiles" in joined.columns else [None] * len(joined)

    cell_states = cell_enc.encode_batch(smiles_list, batch_id=0)
    dmso_tensor = torch.tensor(dmso_mean)

    with torch.no_grad():
        emb                        = layer1(torch.tensor(X))
        aux_zeros                  = torch.zeros(len(X), N_AUX_TARGET)
        pred_expr, pred_var, progs = model(emb, aux_zeros, cell_states, dmso_tensor)

    pred_np  = pred_expr.numpy()
    var_np   = pred_var.numpy()
    progs_np = progs.numpy()
    comp_ids = test_df["compound"].tolist()

    # Long format: one row per (compound, gene_id)
    records = [
        (comp_ids[i], gene, float(pred_np[i, j]))
        for i in range(len(comp_ids))
        for j, gene in enumerate(OUTPUT_GENES)
    ]
    out = pd.DataFrame(records, columns=[vcpi.COMPOUND_COL, vcpi.GENE_COL, vcpi.PRED_COL])
    out.to_parquet(run_dir / "predictions.parquet", index=False)
    print(f"Wrote {run_dir / 'predictions.parquet'}  ({len(out):,} rows)")

    # Save uncertainty + program activations for Layer 4 interpretation
    np.save(run_dir / "uncertainty.npy",         var_np)
    np.save(run_dir / "program_activations.npy", progs_np)
    pd.DataFrame(progs_np, index=comp_ids).to_csv(run_dir / "program_activations.csv")
    return out
