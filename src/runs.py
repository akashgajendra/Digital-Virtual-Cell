"""Run directory helpers — create and discover timestamped run folders."""

from datetime import datetime
from pathlib import Path

from src.config import RUNS_DIR


def make_run_dir() -> Path:
    run_dir = RUNS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def latest_run_dir() -> Path:
    dirs = sorted(RUNS_DIR.glob("*"), key=lambda p: p.name)
    if not dirs:
        raise FileNotFoundError(f"No runs in {RUNS_DIR}/. Run --train first.")
    return dirs[-1]
