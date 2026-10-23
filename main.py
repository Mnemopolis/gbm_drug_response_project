"""
Unified CLI entry point delegating to `src.workflow.run_master_workflow`.

Usage:
    python main.py --stage all --device auto
    python main.py --stage data
    python main.py --stage train --seeds 5 --epochs 100 --device cuda --require-gpu
    python main.py --stage analyze
"""

import argparse

from src.config import EARLY_STOPPING_PATIENCE, MAX_EPOCHS, N_MODEL_SEEDS
from src.workflow import run_master_workflow


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="GBM Drug Response Prediction & OOD Generalization Master CLI"
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["all", "data", "baselines", "train", "uncertainty", "analyze"],
        default="all",
        help="Pipeline stage to execute (default: all).",
    )
    parser.add_argument(
        "--seeds", type=int, default=N_MODEL_SEEDS, help="Random seeds per split."
    )
    parser.add_argument(
        "--epochs", type=int, default=MAX_EPOCHS, help="Max training epochs."
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=EARLY_STOPPING_PATIENCE,
        help="Early stopping patience.",
    )
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"]
    )
    parser.add_argument(
        "--require-gpu", action="store_true", help="Fail if no GPU is available."
    )
    return parser.parse_args()


def main() -> None:
    """Invoke the master workflow with parsed CLI arguments."""
    args = parse_args()
    run_master_workflow(
        stage=args.stage,
        n_seeds=args.seeds,
        epochs=args.epochs,
        patience=args.patience,
        prefer_device=args.device,
        require_gpu=args.require_gpu,
    )


if __name__ == "__main__":
    main()
