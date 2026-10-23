"""
Uncertainty & Multi-Seed Stability Workflow Orchestrator.

Single Responsibility Principle (SRP):
Executes multi-seed stability analysis, diagnoses collapsed seeds on Leave-Cell-Out,
and quantifies OOD prediction shrinkage toward the prior mean.
"""

from typing import Any

import pandas as pd

from src.uncertainty import (
    compare_cell_performance_between_seeds,
    summarize_prediction_shrinkage_and_bias,
    summarize_seed_stability_across_splits,
)


def run_uncertainty_workflow(
    results: dict[str, Any],
    preds_store: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """Run multi-seed stability and prediction shrinkage diagnostics across all splits."""
    print("\n" + "=" * 70)
    print("WORKFLOW 3: Multi-Seed Uncertainty & Prediction Shrinkage Diagnostics")
    print("=" * 70)

    stability_df = summarize_seed_stability_across_splits(results)
    print("\n--- Multi-Seed Training Stability Across Splits ---")
    print(stability_df.to_string(index=False))

    shrinkage_df = summarize_prediction_shrinkage_and_bias(preds_store)
    print("\n--- Prediction Variance Shrinkage (pred_std / true_std) & Bias ---")
    print(shrinkage_df.to_string(index=False))

    outputs: dict[str, pd.DataFrame] = {
        "seed_stability": stability_df,
        "prediction_shrinkage": shrinkage_df,
    }

    if "leave_cell" in results and "seed_runs" in results["leave_cell"]:
        seed_runs = results["leave_cell"]["seed_runs"]
        if len(seed_runs) >= 3:
            test_df = preds_store["leave_cell"]["test_df"]
            collapse_df = compare_cell_performance_between_seeds(
                preds_seed_a=seed_runs[0]["test_preds"],
                preds_seed_b=seed_runs[2]["test_preds"],
                trues=seed_runs[0]["test_trues"],
                cell_ids=test_df["ModelID"].values,
                seed_a_label="seed_0",
                seed_b_label="seed_2",
            )
            print(
                "\n--- Leave-Cell-Out Seed 0 vs. Seed 2 Per-Cell Collapse Diagnosis ---"
            )
            print(collapse_df.to_string(index=False))
            outputs["leave_cell_seed_collapse"] = collapse_df

    return outputs
