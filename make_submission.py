import argparse
import csv
from pathlib import Path


SUBMISSION_COLUMNS = ["compound", "gene_id", "predicted_expression"]
PARQUET_CHUNK_SIZE = 100_000


def latest_run_dir(runs_dir: Path) -> Path:
    runs = sorted(path for path in runs_dir.glob("*") if path.is_dir())
    if not runs:
        raise FileNotFoundError(f"No run directories found under {runs_dir}.")
    return runs[-1]


def read_compounds(path: Path) -> list[str]:
    with path.open(newline="") as f:
        return [row["compound"] for row in csv.DictReader(f)]


def read_genes(path: Path) -> list[str]:
    with path.open() as f:
        return [line.strip() for line in f if line.strip()]


def _write_parquet_chunk(writer, chunk: list[dict]):
    import pyarrow as pa

    table = pa.table(
        {
            "compound": [row["compound"] for row in chunk],
            "gene_id": [row["gene_id"] for row in chunk],
            "predicted_expression": [float(row["predicted_expression"]) for row in chunk],
        },
        schema=pa.schema(
            [
                pa.field("compound", pa.string()),
                pa.field("gene_id", pa.string()),
                pa.field("predicted_expression", pa.float64()),
            ]
        ),
    )
    writer.write_table(table)


def make_submission(
    run_dir: Path,
    test_compounds_path: Path,
    output_path: Path | None = None,
    parquet_output_path: Path | None = None,
) -> tuple[Path, Path]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    predictions_path = run_dir / "predictions.csv"
    genes_path = run_dir / "top_genes.txt"
    if output_path is None:
        output_path = run_dir / "submission.csv"
    if parquet_output_path is None:
        parquet_output_path = output_path.with_suffix(".parquet")

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"No predictions file found at {predictions_path}. "
            "Run model.py --predict first."
        )
    if not genes_path.exists():
        raise FileNotFoundError(f"No gene list found at {genes_path}.")

    test_compounds = read_compounds(test_compounds_path)
    genes = read_genes(genes_path)
    required = SUBMISSION_COLUMNS

    rows = 0
    compounds = set()
    gene_ids = set()
    null_predictions = 0
    negative_predictions = 0
    parquet_schema = pa.schema(
        [
            pa.field("compound", pa.string()),
            pa.field("gene_id", pa.string()),
            pa.field("predicted_expression", pa.float64()),
        ]
    )
    parquet_writer = None
    parquet_chunk = []

    try:
        with predictions_path.open(newline="") as src, output_path.open("w", newline="") as dst:
            reader = csv.DictReader(src)
            missing = [col for col in required if col not in (reader.fieldnames or [])]
            if missing:
                raise ValueError(f"{predictions_path} is missing required columns: {missing}")

            csv_writer = csv.DictWriter(dst, fieldnames=required)
            csv_writer.writeheader()
            parquet_writer = pq.ParquetWriter(parquet_output_path, parquet_schema)
            for row in reader:
                value = row["predicted_expression"]
                if value == "":
                    null_predictions += 1
                else:
                    try:
                        if float(value) < 0:
                            negative_predictions += 1
                    except ValueError:
                        null_predictions += 1

                out_row = {col: row[col] for col in required}
                csv_writer.writerow(out_row)
                parquet_chunk.append(out_row)
                if len(parquet_chunk) >= PARQUET_CHUNK_SIZE:
                    _write_parquet_chunk(parquet_writer, parquet_chunk)
                    parquet_chunk.clear()

                rows += 1
                compounds.add(row["compound"])
                gene_ids.add(row["gene_id"])

            if parquet_chunk:
                _write_parquet_chunk(parquet_writer, parquet_chunk)
                parquet_chunk.clear()
    finally:
        if parquet_writer is not None:
            parquet_writer.close()

    expected_rows = len(test_compounds) * len(genes)
    print(f"Wrote CSV: {output_path}")
    print(f"Wrote Parquet: {parquet_output_path}")
    print(f"Rows: {rows:,}")
    print(f"Expected rows: {expected_rows:,}")
    print(f"Compounds: {len(compounds):,} / expected {len(test_compounds):,}")
    print(f"Genes: {len(gene_ids):,} / expected {len(genes):,}")
    print(f"Null/non-numeric predictions: {null_predictions:,}")
    print(f"Negative predictions: {negative_predictions:,}")

    if rows != expected_rows:
        raise ValueError("Submission row count does not match test_compounds x genes.")
    if len(compounds) != len(test_compounds):
        raise ValueError("Submission compound count does not match test_compounds.csv.")
    if len(gene_ids) != len(genes):
        raise ValueError("Submission gene count does not match top_genes.txt.")
    if null_predictions:
        raise ValueError("Submission contains null or non-numeric predictions.")
    if negative_predictions:
        raise ValueError("Submission contains negative predictions.")

    return output_path, parquet_output_path


def main():
    parser = argparse.ArgumentParser(description="Create contest submission CSV and Parquet files from predictions.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run directory containing predictions.csv. Defaults to the latest run.",
    )
    parser.add_argument(
        "--test-compounds",
        type=Path,
        default=Path("datasets/test_compounds.csv"),
        help="Path to test_compounds.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <run-dir>/submission.csv.",
    )
    parser.add_argument(
        "--parquet-output",
        type=Path,
        default=None,
        help="Output Parquet path. Defaults to the CSV output path with .parquet suffix.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir or latest_run_dir(Path("runs"))
    make_submission(run_dir, args.test_compounds, args.output, args.parquet_output)


if __name__ == "__main__":
    main()
