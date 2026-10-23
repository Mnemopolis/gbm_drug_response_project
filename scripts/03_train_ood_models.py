"""
Step 03: Train Two-Tower BaselineMLP across Random, Leave-Cell-Out, and Scaffold
splits (x 5 seeds) on GPU/CPU and save OOD_results.pkl & OOD_preds.pkl.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EARLY_STOPPING_PATIENCE, MAX_EPOCHS, N_MODEL_SEEDS, SPLIT_NAMES
from src.mlp_workflow import run_ood_mlp_workflow

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 03: Train Multi-Split OOD Models"
    )
    parser.add_argument("--seeds", type=int, default=N_MODEL_SEEDS)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"]
    )
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()

    run_ood_mlp_workflow(
        split_names=SPLIT_NAMES,
        n_seeds=args.seeds,
        epochs=args.epochs,
        patience=args.patience,
        prefer_device=args.device,
        require_gpu=args.require_gpu,
        save_results=True,
    )
