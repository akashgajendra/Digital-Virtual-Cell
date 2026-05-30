"""
Layer 4 — Gene program interpretation.

Takes gene_program_activations saved by predict.py and maps them to
biological pathway names via GSEA.  Never modifies predictions.parquet.

Install gseapy to enable pathway naming: uv add gseapy
"""

import numpy as np
import pandas as pd

from src.config import OUTPUT_GENES
from src.fetch_external import load_gene_symbols


def interpret_programs(
    activations:  np.ndarray,   # (n_compounds, 128)
    nmf_H:        np.ndarray,   # (128, n_genes)
    top_programs: int = 5,
    top_genes:    int = 20,
) -> pd.DataFrame:
    """
    For each compound, identify the most-activated gene programs and the
    top-loaded genes in each.  Returns a DataFrame for inspection/reporting.
    Shows HGNC symbols if gene_symbols.csv has been fetched; otherwise ENSEMBL IDs.
    """
    symbols = load_gene_symbols()  # {} if not yet fetched

    def _name(gene_id: str) -> str:
        return symbols.get(gene_id, gene_id)

    records = []
    for i, acts in enumerate(activations):
        for prog_id in np.argsort(acts)[::-1][:top_programs]:
            loadings       = nmf_H[prog_id]
            top_gene_idx   = np.argsort(loadings)[::-1][:top_genes]
            top_gene_names = [_name(OUTPUT_GENES[j]) for j in top_gene_idx]
            records.append({
                "compound_idx":     i,
                "program_id":       int(prog_id),
                "activation_score": float(acts[prog_id]),
                "top_genes":        ", ".join(top_gene_names),
            })
    return pd.DataFrame(records)


def run_gsea_on_programs(
    nmf_H:       np.ndarray,
    program_ids: list[int] | None = None,
) -> dict | None:
    """
    Name each gene program by running GSEA prerank on NMF gene loadings
    against MSigDB Hallmark gene sets.
    Returns {program_id: [top_pathway_names]}.
    """
    try:
        import gseapy as gp
    except ImportError:
        print("gseapy not installed. Run: uv add gseapy")
        return None

    program_ids = program_ids or list(range(nmf_H.shape[0]))
    results: dict[int, list[str]] = {}
    for pid in program_ids:
        ranked = pd.Series(nmf_H[pid], index=OUTPUT_GENES).sort_values(ascending=False)
        res    = gp.prerank(
            rnk=ranked, gene_sets="MSigDB_Hallmark_2020",
            outdir=None, verbose=False, permutation_num=100, seed=42,
        )
        top = res.res2d.head(3)["Term"].tolist() if not res.res2d.empty else ["(no enrichment)"]
        results[pid] = top
        print(f"  Program {pid:3d}: {', '.join(top)}")
    return results
