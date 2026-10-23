"""
Baseline Evaluation Workflow Module.

Single Responsibility Principle (SRP):
Orchestrates statistical baseline evaluations (Cell-only, Drug-only, Additive,
Within-Cell Drug-only, and Nearest-Drug Morgan FP similarity) across all splits.
"""

from typing import Any

import pandas as pd

from src.baselines import (
    compute_drug_only_predictions,
    compute_within_cell_spearman,
    evaluate_all_statistical_baselines,
    evaluate_nearest_drug_scaffold_baseline,
)
from src.config import SPLIT_NAMES, get_split_data_paths
from src.splitting import partition_train_validation


def run_baseline_workflow() -> dict[str, dict[str, Any]]:
    """Run all non-deep-learning statistical and chemical similarity baselines across splits."""
    print("=" * 70)
    print("WORKFLOW: Evaluating Statistical & Chemical Similarity Baselines")
    print("=" * 70)

    summary: dict[str, dict[str, Any]] = {}
    for split in SPLIT_NAMES:
        paths = get_split_data_paths(split)
        trainval_df = pd.read_pickle(paths["train"])
        test_df = pd.read_pickle(paths["test"]).reset_index(drop=True)
        train_df, _ = partition_train_validation(trainval_df)

        stats = evaluate_all_statistical_baselines(train_df, test_df)
        pred_drug = compute_drug_only_predictions(train_df, test_df)
        mask = pred_drug.notna()
        wc_do_mean, wc_do_std, wc_do_n = compute_within_cell_spearman(
            pred_drug[mask].values,
            test_df.loc[mask, "LN_IC50"].values,
            test_df.loc[mask, "ModelID"].values,
        )

        nn_sp = float("nan")
        if split == "scaffold":
            nn_sp = evaluate_nearest_drug_scaffold_baseline(trainval_df, test_df)

        summary[split] = {
            **stats,
            "within_cell_drug_only_mean": wc_do_mean,
            "within_cell_drug_only_std": wc_do_std,
            "within_cell_drug_only_n": wc_do_n,
            "nearest_drug_spearman": nn_sp,
        }
        print(
            f"  [{split:<10s}] Cell={stats['cell_only']:.4f} | "
            f"Drug={stats['drug_only']:.4f} | Additive={stats['additive']:.4f} | "
            f"WC-Drug={wc_do_mean:.4f} | Nearest-Drug={nn_sp:.4f}"
        )

    return summary
