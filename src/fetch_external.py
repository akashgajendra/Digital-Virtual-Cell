"""
Fetch and cache external data needed by Layers 3 and 4.
Run this once before training.

    uv run python -m src.fetch_external

Saves to datasets/:
    chembl_targets.parquet   — InChIKey × 64 target binary matrix  (Layer 3 aux branch)
    gene_symbols.csv         — ENSEMBL ID → HGNC symbol            (Layer 4 readability)

MSigDB Hallmark is cached automatically by gseapy on first use — no manual step needed.
"""

import glob
import time
from pathlib import Path

import numpy as np
import pandas as pd
import vcpi_prediction_contest as vcpi

from src.config import DATASETS_DIR, N_AUX_TARGET, OUTPUT_GENES

CHEMBL_TARGETS_PATH = DATASETS_DIR / "chembl_targets.parquet"
GENE_SYMBOLS_PATH   = DATASETS_DIR / "gene_symbols.csv"

_BATCH   = 50    # ChEMBL recommended batch size
_SLEEP   = 0.3   # seconds between batches — polite rate limiting


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _discover_compounds() -> pd.DataFrame:
    matches = sorted(glob.glob(str(DATASETS_DIR / "compounds-*.csv")))
    if not matches:
        raise FileNotFoundError(f"No compounds file in {DATASETS_DIR}/")
    return pd.read_csv(matches[0])


def _all_inchikeys() -> pd.DataFrame:
    """
    Collect InChIKeys for training compounds (from compounds CSV)
    and test compounds (from vcpi package). Returns (id, inchi_key) DataFrame
    where id = compound UUID for training, user_compound_id for test.
    """
    train = _discover_compounds()[["compound", "inchi_key"]].dropna()
    train = train.rename(columns={"compound": "id"})

    test = vcpi.load_test_compounds()[["compound", "inchi_key"]].dropna()
    test = test.rename(columns={"compound": "id"})

    return pd.concat([train, test], ignore_index=True).drop_duplicates("inchi_key")


# ---------------------------------------------------------------------------
# 1. ChEMBL targets  (Layer 3 auxiliary branch)
# ---------------------------------------------------------------------------

def fetch_chembl_targets(force: bool = False) -> pd.DataFrame:
    """
    For every compound (training + test), look up which proteins it binds in ChEMBL.
    Builds a binary matrix: rows = InChIKey, columns = top-64 targets by compound coverage.

    Returns and saves datasets/chembl_targets.parquet.
    Compounds with no ChEMBL entry stay as all-zeros (same as current baseline).
    """
    if CHEMBL_TARGETS_PATH.exists() and not force:
        print(f"ChEMBL targets already cached at {CHEMBL_TARGETS_PATH}  (pass force=True to re-fetch)")
        return pd.read_parquet(CHEMBL_TARGETS_PATH)

    try:
        from chembl_webresource_client.new_client import new_client
    except ImportError:
        raise ImportError("Run: uv add chembl_webresource_client")

    all_cmpds = _all_inchikeys()
    inchikeys  = all_cmpds["inchi_key"].tolist()
    print(f"Fetching ChEMBL data for {len(inchikeys)} unique compounds ...")

    molecule = new_client.molecule
    activity = new_client.activity

    # ------------------------------------------------------------------
    # Step 1: InChIKey → ChEMBL molecule ID  (batched)
    # ------------------------------------------------------------------
    ik_to_chembl: dict[str, str] = {}
    for i in range(0, len(inchikeys), _BATCH):
        batch = inchikeys[i : i + _BATCH]
        try:
            mols = molecule.filter(
                molecule_structures__standard_inchi_key__in=batch
            ).only(["molecule_chembl_id", "molecule_structures"])
            for mol in mols:
                ik = mol.get("molecule_structures", {}).get("standard_inchi_key")
                if ik:
                    ik_to_chembl[ik] = mol["molecule_chembl_id"]
        except Exception as e:
            print(f"  Warning: InChIKey batch {i} failed ({e})")
        time.sleep(_SLEEP)
        if i % 500 == 0:
            print(f"  InChIKey lookup: {i}/{len(inchikeys)} done")

    print(f"Found ChEMBL IDs for {len(ik_to_chembl)}/{len(inchikeys)} compounds")

    if not ik_to_chembl:
        print("No ChEMBL matches — saving zeros matrix")
        return _save_zeros(all_cmpds)

    # ------------------------------------------------------------------
    # Step 2: ChEMBL ID → binding activities  (batched)
    # ------------------------------------------------------------------
    chembl_ids = list(ik_to_chembl.values())
    rows: list[dict] = []

    for i in range(0, len(chembl_ids), _BATCH):
        batch = chembl_ids[i : i + _BATCH]
        try:
            acts = activity.filter(
                molecule_chembl_id__in=batch,
                assay_type="B",
                standard_type__in=["IC50", "Ki", "Kd", "EC50"],
            ).only(["molecule_chembl_id", "target_chembl_id"])
            rows.extend(list(acts))
        except Exception as e:
            print(f"  Warning: activity batch {i} failed ({e})")
        time.sleep(_SLEEP)
        if i % 500 == 0:
            print(f"  Activity lookup: {i}/{len(chembl_ids)} done")

    if not rows:
        print("No activity data retrieved — saving zeros matrix")
        return _save_zeros(all_cmpds)

    acts_df = pd.DataFrame(rows).dropna()
    print(f"Retrieved {len(acts_df):,} activity records across {acts_df['target_chembl_id'].nunique()} targets")

    # ------------------------------------------------------------------
    # Step 3: Select top N_AUX_TARGET targets by compound coverage
    # ------------------------------------------------------------------
    chembl_to_ik = {v: k for k, v in ik_to_chembl.items()}
    acts_df["inchi_key"] = acts_df["molecule_chembl_id"].map(chembl_to_ik)
    acts_df = acts_df.dropna(subset=["inchi_key", "target_chembl_id"])

    top_targets = (
        acts_df.groupby("target_chembl_id")["inchi_key"]
        .nunique()
        .nlargest(N_AUX_TARGET)
        .index.tolist()
    )
    print(f"Top {N_AUX_TARGET} targets selected")

    # ------------------------------------------------------------------
    # Step 4: Build binary (inchi_key × target) matrix
    # ------------------------------------------------------------------
    pivot = (
        acts_df[acts_df["target_chembl_id"].isin(top_targets)]
        .assign(active=1.0)
        .pivot_table(index="inchi_key", columns="target_chembl_id", values="active", fill_value=0.0)
        .reindex(columns=top_targets, fill_value=0.0)
    )
    # Ensure every compound is present (zeros if not in ChEMBL)
    pivot = pivot.reindex(inchikeys, fill_value=0.0).fillna(0.0).astype(np.float32)

    pivot.to_parquet(CHEMBL_TARGETS_PATH)
    print(f"Saved → {CHEMBL_TARGETS_PATH}  {pivot.shape}")
    return pivot


def _save_zeros(all_cmpds: pd.DataFrame) -> pd.DataFrame:
    cols  = [f"target_{i}" for i in range(N_AUX_TARGET)]
    zeros = pd.DataFrame(0.0, index=all_cmpds["inchi_key"], columns=cols, dtype=np.float32)
    zeros.to_parquet(CHEMBL_TARGETS_PATH)
    return zeros


def load_chembl_targets() -> pd.DataFrame | None:
    """Load cached ChEMBL target matrix. Returns None if not yet fetched."""
    if not CHEMBL_TARGETS_PATH.exists():
        return None
    return pd.read_parquet(CHEMBL_TARGETS_PATH)


# ---------------------------------------------------------------------------
# 2. Gene symbols  (Layer 4 readability)
# ---------------------------------------------------------------------------

def fetch_gene_symbols(force: bool = False) -> pd.DataFrame:
    """
    Map the 12,995 ENSEMBL gene IDs to HGNC symbols via mygene.info.
    Falls back to the ENSEMBL ID itself when no symbol is found.

    Returns and saves datasets/gene_symbols.csv.
    """
    if GENE_SYMBOLS_PATH.exists() and not force:
        print(f"Gene symbols already cached at {GENE_SYMBOLS_PATH}  (pass force=True to re-fetch)")
        return pd.read_csv(GENE_SYMBOLS_PATH)

    try:
        import mygene
    except ImportError:
        raise ImportError("Run: uv add mygene")

    print(f"Fetching symbols for {len(OUTPUT_GENES)} genes via mygene.info ...")
    mg = mygene.MyGeneInfo()
    results = mg.querymany(
        OUTPUT_GENES,
        scopes="ensembl.gene",
        fields="symbol",
        species="human",
        as_dataframe=True,
        verbose=False,
    )

    df = (
        results[["symbol"]]
        .reset_index()
        .rename(columns={"query": "gene_id"})
        .drop_duplicates("gene_id")
    )
    df["symbol"] = df["symbol"].fillna(df["gene_id"])  # fallback to ENSEMBL ID

    df.to_csv(GENE_SYMBOLS_PATH, index=False)
    print(f"Saved → {GENE_SYMBOLS_PATH}  ({len(df)} genes, "
          f"{df['symbol'].ne(df['gene_id']).sum()} with HGNC symbols)")
    return df


def load_gene_symbols() -> dict[str, str]:
    """
    Load cached ENSEMBL → symbol mapping as a dict.
    Returns identity mapping (gene_id → gene_id) if not yet fetched.
    """
    if not GENE_SYMBOLS_PATH.exists():
        return {}
    df = pd.read_csv(GENE_SYMBOLS_PATH)
    return dict(zip(df["gene_id"], df["symbol"]))


# ---------------------------------------------------------------------------
# 3. MSigDB Hallmark  (Layer 4 GSEA)
# ---------------------------------------------------------------------------

def prefetch_msigdb():
    """
    Pre-download MSigDB Hallmark gene sets so the first --interpret run is fast.
    gseapy caches this locally after the first download.
    """
    try:
        import gseapy as gp
    except ImportError:
        raise ImportError("Run: uv add gseapy")

    print("Pre-fetching MSigDB Hallmark gene sets ...")
    gp.get_library("MSigDB_Hallmark_2020")
    print("MSigDB Hallmark cached.")


# ---------------------------------------------------------------------------
# Entry point — run all three in sequence
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fetch external data for Layers 3 and 4")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if cache exists")
    args = parser.parse_args()

    print("=" * 50)
    print("1 / 3  ChEMBL targets")
    print("=" * 50)
    fetch_chembl_targets(force=args.force)

    print("\n" + "=" * 50)
    print("2 / 3  Gene symbols")
    print("=" * 50)
    fetch_gene_symbols(force=args.force)

    print("\n" + "=" * 50)
    print("3 / 3  MSigDB Hallmark")
    print("=" * 50)
    prefetch_msigdb()

    print("\nAll done. Run: uv run python model.py --train")
