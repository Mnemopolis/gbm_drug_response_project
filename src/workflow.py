"""
Master End-to-End Pipeline Orchestrator (`workflow.py`).

Single Responsibility Principle (SRP):
Connects `split_workflow`, `baseline_workflow`, `mlp_workflow`,
`uncertainty_workflow`, and `results_analysis` into a single cohesive entry point.
"""

from src.baseline_workflow import run_baseline_workflow
from src.config import EARLY_STOPPING_PATIENCE, MAX_EPOCHS, N_MODEL_SEEDS, SPLIT_NAMES
from src.mlp_workflow import (
    load_saved_results_and_predictions,
    run_ood_mlp_workflow,
)
from src.results_analysis import run_results_analysis_workflow
from src.split_workflow import run_split_workflow
from src.uncertainty_workflow import run_uncertainty_workflow


def run_master_workflow(
    stage: str = "all",
    n_seeds: int = N_MODEL_SEEDS,
    epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
    prefer_device: str = "auto",
    require_gpu: bool = False,
) -> None:
    """Execute the requested pipeline stage(s) from start to finish."""
    if stage in ("all", "data"):
        run_split_workflow(save_splits=True)

    if stage in ("all", "baselines"):
        run_baseline_workflow()

    if stage in ("all", "train"):
        results, preds_store = run_ood_mlp_workflow(
            split_names=SPLIT_NAMES,
            n_seeds=n_seeds,
            epochs=epochs,
            patience=patience,
            prefer_device=prefer_device,
            require_gpu=require_gpu,
            save_results=True,
        )
    elif stage in ("uncertainty", "analyze"):
        results, preds_store = load_saved_results_and_predictions()
    else:
        return

    if stage in ("all", "train", "uncertainty", "analyze"):
        run_uncertainty_workflow(results, preds_store)

    if stage in ("all", "train", "analyze"):
        run_results_analysis_workflow(results, preds_store)
