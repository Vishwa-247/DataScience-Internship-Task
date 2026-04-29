"""Phase 7 — CLI: ``python -m src.training.cli --states all``.

Usage::

    python -m src.training.cli --states all
    python -m src.training.cli --states California Texas --n-jobs 2
"""

from __future__ import annotations

import argparse
import sys
import time

from src.data.loader import load_config, load_sales
from src.data.preprocessor import Preprocessor
from src.training.pipeline import TrainingPipeline
from src.utils.logging import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def _print_results_table(results: list[dict]) -> None:
    """Print a compact summary table to stdout."""
    header = f"{'State':<20} {'Top-2':<25} {'Weights':<20} " + " ".join(
        f"{'RMSE_' + m:<16}" for m in ["arima", "prophet", "xgboost", "lstm"]
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    for r in sorted(results, key=lambda x: x["state"]):
        sel = "+".join(r["selected"])
        wts = " / ".join(f"{r['weights'][m]:.2f}" for m in r["selected"])
        rmse_cols = " ".join(
            f"{r['mean_rmse'].get(m, float('inf')):>16,.0f}"
            for m in ["arima", "prophet", "xgboost", "lstm"]
        )
        print(f"{r['state']:<20} {sel:<25} {wts:<20} {rmse_cols}")

    print("=" * len(header) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="src.training.cli",
        description="Train all models for selected states with walk-forward CV.",
    )
    parser.add_argument(
        "--states",
        nargs="+",
        default=["all"],
        help='State names or "all" (default: all)',
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel workers (-1 = all cores, default: 1 for safety)",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    logger.info("Loading and preprocessing data …")
    df = load_sales(config=config)
    state_series = Preprocessor(config).transform(df)

    # Resolve --states
    if args.states == ["all"]:
        states = None  # pipeline trains all
    else:
        states = args.states

    t0 = time.time()
    pipeline = TrainingPipeline(config)
    version, results = pipeline.run(state_series, states=states, n_jobs=args.n_jobs)
    elapsed = time.time() - t0

    _print_results_table(results)
    print(f"Registry version: {version}")
    print(f"Total time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"States trained: {len(results)}")


if __name__ == "__main__":
    main()
