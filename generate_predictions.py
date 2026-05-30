from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rdkit import Chem
from rdkit.Chem import Descriptors

from vcpi_prediction_contest import load_test_compounds
from model import load_trained_model


# -----------------------
# Load
# -----------------------
tc = load_test_compounds()
model, scaler, top_genes, run_dir = load_trained_model()


# -----------------------
# MUST MATCH TRAINING FEATURES (8 dims)
# -----------------------
def featurize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(8, dtype=np.float32)

    return np.array([
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumHDonors(mol),
        mol.GetNumAtoms(),
        mol.GetNumBonds()
    ], dtype=np.float32)


# -----------------------
# Build X (8 features ONLY)
# -----------------------
X = np.vstack([featurize(smi) for smi in tc["smiles"]]).astype(np.float32)

X = scaler.transform(X)

print("Feature shape:", X.shape)


# -----------------------
# Inference
# -----------------------
model.eval()
with torch.no_grad():
    preds = model(torch.tensor(X)).cpu().numpy()


# -----------------------
# Output
# -----------------------
records = []
for i, row in tc.iterrows():
    for j, gene in enumerate(top_genes):
        records.append((row["compound"], gene, float(preds[i, j])))

submission = pd.DataFrame(records, columns=[
    "compound", "gene_id", "predicted_lfc"
])

submission.to_csv("predictions.csv", index=False)
print("Saved predictions.csv")

out_path = Path("datasets/test_compounds.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)
tc.to_csv(out_path, index=False)
print(f"Wrote {out_path}")
