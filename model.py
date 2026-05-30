"""
Drug-seq perturbation model — entry point.

Usage:
    uv run python model.py --train
    uv run python model.py --predict
    uv run python model.py --train --predict
    uv run python model.py --interpret --run-dir runs/20260530_143022
"""

import argparse
from pathlib import Path

from src.interpret import interpret_programs, run_gsea_on_programs
from src.predict import predict
from src.runs import latest_run_dir
from src.train import train

import numpy as np


def interpret(run_dir: Path | None = None):
    """Layer 4: name gene programs via GSEA. Run after --predict."""
    if run_dir is None:
        run_dir = latest_run_dir()

    act_path = run_dir / "program_activations.npy"
    if not act_path.exists():
        raise FileNotFoundError("Run --predict first to generate program_activations.npy")

    activations = np.load(act_path)
    nmf_H       = np.load(run_dir / "nmf_H.npy")

    print(f"\n=== Gene program summary ({activations.shape[0]} compounds) ===")
    df = interpret_programs(activations, nmf_H, top_programs=5, top_genes=10)
    df.to_csv(run_dir / "program_interpretation.csv", index=False)
    print(df.to_string(index=False))

    print("\n=== GSEA pathway names for top 5 programs by mean activation ===")
    top5 = np.argsort(activations.mean(axis=0))[::-1][:5].tolist()
    run_gsea_on_programs(nmf_H, program_ids=top5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drug-seq perturbation model")
    parser.add_argument("--train",     action="store_true", help="Train the model")
    parser.add_argument("--predict",   action="store_true", help="Predict on test compounds")
    parser.add_argument("--interpret", action="store_true", help="Layer 4: name gene programs")
    parser.add_argument("--run-dir",   metavar="DIR",       help="Run dir to load from (default: latest)")
    args = parser.parse_args()

    if not any([args.train, args.predict, args.interpret]):
        parser.print_help()

    run_dir = Path(args.run_dir) if args.run_dir else None
    if args.train:
        run_dir = train()
    if args.predict:
        predict(run_dir=run_dir)
    if args.interpret:
        interpret(run_dir=run_dir)
