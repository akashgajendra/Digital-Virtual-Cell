"""Training pipeline."""

import pickle
from pathlib import Path

import numpy as np
import torch
import vcpi_prediction_contest as vcpi
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.cell_state import CellStateEncoder
from src.config import (
    BATCH_SIZE,
    COMPOUND_FEATURE_COLS,
    EPOCHS,
    LR,
    N_OUTPUT_GENES,
    N_PROGRAMS,
    OUTPUT_GENES,
)
from src.data import (
    PerturbationDataset,
    build_expression_table,
    compute_dmso_baseline,
    compute_gene_weights,
    compute_nmf_factors,
    load_raw_data,
)
from src.fetch_external import load_chembl_targets
from src.placeholder import PlaceholderLayer1
from src.fusion import FusionModel
from src.loss import kl_variance_loss, wmse_loss
from src.runs import make_run_dir


def train(epochs: int = EPOCHS, lr: float = LR) -> Path:
    run_dir = make_run_dir()
    print(f"Run directory: {run_dir}\n")

    counts, meta, compounds_df = load_raw_data()

    # Expression table (official normalizer, QC-filtered)
    print("\nBuilding expression table...")
    expr_long = build_expression_table(counts, meta)
    expr_wide = (
        expr_long
        .pivot(index=vcpi.COMPOUND_COL, columns=vcpi.GENE_COL, values=vcpi.EXPRESSION_COL)
        .reindex(columns=OUTPUT_GENES)
        .dropna(how="all")
        .astype(np.float32)
    )
    print(f"Expression matrix: {expr_wide.shape}  (compounds × genes)")

    # DMSO baseline and Mejia gene weights
    print("\nComputing DMSO baseline and gene weights...")
    dmso_mean    = compute_dmso_baseline(counts, meta)
    gene_weights = compute_gene_weights(counts, meta).reindex(OUTPUT_GENES).fillna(0).values.astype(np.float32)

    # NMF factors to initialise the gene decoder
    _, nmf_H = compute_nmf_factors(expr_wide, N_PROGRAMS)

    # Layer 2: fit cell state encoder on DMSO cells
    print("\nFitting CellStateEncoder (Layer 2)...")
    cell_enc = CellStateEncoder()
    cell_enc.fit(counts, meta)

    # Align compound features with expression rows (both indexed by UUID)
    compounds_df = compounds_df.set_index("compound")
    common       = expr_wide.index.intersection(compounds_df.index)
    print(f"\nCompounds aligned: {len(common)}")

    expr_aligned = expr_wide.loc[common].values.astype(np.float32)
    feats_raw    = compounds_df.loc[common, COMPOUND_FEATURE_COLS].fillna(0).values.astype(np.float32)
    smiles_list  = compounds_df.loc[common, "smiles"].tolist() if "smiles" in compounds_df.columns else [None] * len(common)

    feat_scaler = StandardScaler()
    feats       = feat_scaler.fit_transform(feats_raw)

    print("Computing cell state vectors...")
    cell_states = cell_enc.encode_batch(smiles_list, batch_id=0).numpy()

    # ChEMBL auxiliary targets — None if not yet fetched (aux branch stays zero)
    chembl_df   = load_chembl_targets()
    inchikeys   = compounds_df.loc[common, "inchi_key"].tolist() if "inchi_key" in compounds_df.columns else None
    aux_targets = None
    if chembl_df is not None and inchikeys is not None:
        aux_targets = chembl_df.reindex(inchikeys).fillna(0).values.astype(np.float32)
        print(f"ChEMBL targets loaded: {aux_targets.shape}")
    else:
        print("ChEMBL targets not found — aux branch will use zeros. Run: uv run python -m src.fetch_external")

    # Dataset + loaders
    dataset = PerturbationDataset(feats, cell_states, expr_aligned, dmso_mean.values, aux_targets)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Layer 1 (placeholder) + Layer 3 model
    layer1 = PlaceholderLayer1(input_dim=len(COMPOUND_FEATURE_COLS))
    model  = FusionModel(n_genes=N_OUTPUT_GENES, n_programs=N_PROGRAMS)
    model.init_with_nmf(nmf_H)

    optimizer = torch.optim.AdamW(
        list(layer1.parameters()) + list(model.parameters()), lr=lr, weight_decay=1e-4
    )
    w_tensor = torch.tensor(gene_weights)
    emp_var  = torch.tensor(
        expr_wide.var(axis=0).reindex(OUTPUT_GENES).fillna(0).values.astype(np.float32)
    )

    print(f"\nTraining {len(dataset)} compounds → {N_OUTPUT_GENES} genes for {epochs} epochs")
    layer1.train(); model.train()

    for epoch in range(1, epochs + 1):
        total = 0.0
        for x_c, x_a, x_s, dmso, y in loader:
            optimizer.zero_grad()
            emb                    = layer1(x_c)
            pred_expr, pred_var, _ = model(emb, x_a, x_s, dmso)
            loss = wmse_loss(pred_expr, y, w_tensor) + 0.1 * kl_variance_loss(pred_var, emp_var)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(x_c)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{epochs}  loss={total/len(dataset):.4f}")

    # Save artifacts
    torch.save({
        "layer1_state": layer1.state_dict(),
        "model_state":  model.state_dict(),
        "input_dim":    len(COMPOUND_FEATURE_COLS),
        "n_genes":      N_OUTPUT_GENES,
        "n_programs":   N_PROGRAMS,
    }, run_dir / "model_checkpoint.pt")

    with open(run_dir / "feat_scaler.pkl", "wb") as f: pickle.dump(feat_scaler, f)
    with open(run_dir / "cell_enc.pkl",    "wb") as f: pickle.dump(cell_enc, f)

    np.save(run_dir / "nmf_H.npy",        nmf_H)
    np.save(run_dir / "dmso_mean.npy",     dmso_mean.values)
    np.save(run_dir / "gene_weights.npy",  gene_weights)

    print(f"\nArtifacts saved to {run_dir}/")
    return run_dir
