"""
Multi-Seed Uncertainty, Training Instability, and Prediction Variance Shrinkage Module.

Single Responsibility Principle (SRP):
Quantifies initialization sensitivity across random seeds (e.g., diagnosing the
Seed 2 early-stopping collapse on Leave-Cell-Out in Notebook 03 Cells 7-8) and
prediction variance shrinkage toward the training mean (Notebook 04 Cell 8).
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.config import SPLIT_NAMES


def summarize_seed_stability_across_splits(results: dict[str, Any]) -> pd.DataFrame:
    """
    Summarize multi-seed Spearman and RMSE mean, standard deviation, min, max,
    and instability flag across splits.
    """
    rows: list[dict[str, Any]] = []
    for split in SPLIT_NAMES:
        if split not in results:
            continue
        sp_vals = np.asarray(results[split]["model_global"]["values"], dtype=np.float64)
        rmse_info = results[split]["model_rmse"]
        rmse_vals = np.asarray(
            rmse_info.get("values", [rmse_info["mean"]]), dtype=np.float64
        )
        rows.append(
            {
                "split": split,
                "n_seeds": len(sp_vals),
                "spearman_mean": float(np.mean(sp_vals)),
                "spearman_std": float(np.std(sp_vals)),
                "spearman_min": float(np.min(sp_vals)),
                "spearman_max": float(np.max(sp_vals)),
                "rmse_mean": float(np.mean(rmse_vals)),
                "rmse_std": float(np.std(rmse_vals)),
                "has_collapsed_seed": bool(
                    np.min(sp_vals) < 0.35 and np.max(sp_vals) > 0.70
                ),
            }
        )
    return pd.DataFrame(rows)


def compare_cell_performance_between_seeds(
    preds_seed_a: np.ndarray,
    preds_seed_b: np.ndarray,
    trues: np.ndarray,
    cell_ids: np.ndarray,
    seed_a_label: str = "seed_0",
    seed_b_label: str = "seed_2",
) -> pd.DataFrame:
    """
    Compare per-cell-line Spearman correlation, RMSE, and mean predicted LN_IC50
    between two random seeds (e.g., normal Seed 0 vs. collapsed Seed 2 on Leave-Cell-Out).
    """
    df = pd.DataFrame(
        {
            "cell": cell_ids,
            "true": trues,
            "pred_a": preds_seed_a,
            "pred_b": preds_seed_b,
        }
    )
    records: list[dict[str, Any]] = []
    for cell_id, group in df.groupby("cell"):
        sp_a = float(spearmanr(group["pred_a"], group["true"]).correlation)
        sp_b = float(spearmanr(group["pred_b"], group["true"]).correlation)
        rmse_a = float(np.sqrt(np.mean((group["pred_a"] - group["true"]) ** 2)))
        rmse_b = float(np.sqrt(np.mean((group["pred_b"] - group["true"]) ** 2)))
        records.append(
            {
                "ModelID": cell_id,
                "n_drugs": len(group),
                "true_mean": float(group["true"].mean()),
                f"{seed_a_label}_spearman": sp_a,
                f"{seed_a_label}_rmse": rmse_a,
                f"{seed_b_label}_spearman": sp_b,
                f"{seed_b_label}_rmse": rmse_b,
                "spearman_drop": sp_a - sp_b,
            }
        )
    return pd.DataFrame(records)


def summarize_prediction_shrinkage_and_bias(
    preds_store: dict[str, Any],
) -> pd.DataFrame:
    """
    Quantify prediction variance shrinkage toward the training mean (`pred_std / true_std`)
    and systematic signed bias (`pred - true`) across splits.
    """
    rows = []
    for split in SPLIT_NAMES:
        if split not in preds_store:
            continue
        p = preds_store[split]
        preds = np.asarray(p["preds"])
        trues = np.asarray(p["trues"])
        err = preds - trues
        pred_std = float(np.std(preds))
        true_std = float(np.std(trues))
        rows.append(
            {
                "split": split,
                "pred_std": pred_std,
                "true_std": true_std,
                "std_ratio": pred_std / max(1e-8, true_std),
                "mean_err": float(np.mean(err)),
                "median_err": float(np.median(err)),
                "frac_abs_err_gt_2": float((np.abs(err) > 2.0).mean()),
            }
        )
    return pd.DataFrame(rows)
