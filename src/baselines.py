"""
Non-deep-learning statistical and chemical similarity baselines and reusable
group-wise regression metrics for diagnosing shortcut learning and OOD generalization.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:

    def cosine_similarity(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Pure NumPy cosine similarity matrix between rows of x and rows of y."""
        x_norm = np.linalg.norm(x, axis=1, keepdims=True)
        y_norm = np.linalg.norm(y, axis=1, keepdims=True)
        x_safe = x / np.where(x_norm < 1e-8, 1.0, x_norm)
        y_safe = y / np.where(y_norm < 1e-8, 1.0, y_norm)
        return x_safe @ y_safe.T


def safe_spearman(preds: np.ndarray | pd.Series, trues: np.ndarray | pd.Series) -> float:
    """Compute Spearman rank correlation with zero-variance and minimum-length guards."""
    p_arr = np.asarray(preds, dtype=np.float64)
    t_arr = np.asarray(trues, dtype=np.float64)
    mask = ~(np.isnan(p_arr) | np.isnan(t_arr))
    p_valid, t_valid = p_arr[mask], t_arr[mask]
    if len(p_valid) < 2 or np.std(p_valid) < 1e-8 or np.std(t_valid) < 1e-8:
        return float("nan")
    return float(spearmanr(p_valid, t_valid).correlation)


def compute_grouped_regression_stats(
    df: pd.DataFrame,
    group_col: str,
    pred_col: str = "pred",
    true_col: str = "true",
    min_n: int = 3,
    include_true_std: bool = False,
) -> pd.DataFrame:
    """
    Reusable group-level regression evaluator computing count `n`, `rmse`, `mean_err`,
    and `spearman` across groups (`DRUG_NAME` or `ModelID`) with constant-input guards.
    """
    base_cols = [group_col, "n", "rmse", "mean_err", "spearman"]
    if include_true_std:
        base_cols.append("true_std")

    rows: list[dict[str, Any]] = []
    for group_key, group in df.groupby(group_col):
        if len(group) < min_n:
            continue
        preds = group[pred_col].to_numpy(dtype=np.float64)
        trues = group[true_col].to_numpy(dtype=np.float64)
        err = preds - trues
        sp_val = safe_spearman(preds, trues) if len(group) >= 3 else float("nan")
        row: dict[str, Any] = {
            group_col: group_key,
            "n": len(group),
            "rmse": float(np.sqrt(np.mean(err**2))),
            "mean_err": float(np.mean(err)),
            "spearman": sp_val if not np.isnan(sp_val) else np.nan,
        }
        if include_true_std:
            row["true_std"] = float(group[true_col].std()) if len(group) > 1 else 0.0
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=base_cols)
    return pd.DataFrame(rows).sort_values("rmse", ascending=False).reset_index(drop=True)


def compute_drug_only_predictions(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> pd.Series:
    """Predict test LN_IC50 using the per-drug mean LN_IC50 observed in the training set."""
    drug_mean_map = train_df.groupby("DRUG_NAME")["LN_IC50"].mean()
    return test_df["DRUG_NAME"].map(drug_mean_map)


def compute_cell_only_predictions(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> pd.Series:
    """Predict test LN_IC50 using the per-cell-line mean LN_IC50 observed in the training set."""
    cell_mean_map = train_df.groupby("ModelID")["LN_IC50"].mean()
    return test_df["ModelID"].map(cell_mean_map)


def compute_additive_predictions(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> pd.Series:
    """Predict test LN_IC50 using an additive main-effects model: drug_mean + cell_mean - global_mean."""
    global_mean = float(train_df["LN_IC50"].mean())
    return (
        compute_drug_only_predictions(train_df, test_df)
        + compute_cell_only_predictions(train_df, test_df)
        - global_mean
    )


def compute_valid_spearman(
    predictions: pd.Series, targets: pd.Series
) -> tuple[float, float]:
    """Compute Spearman rank correlation and valid coverage fraction for non-NaN predictions."""
    valid_mask = predictions.notna() & targets.notna()
    coverage = float(valid_mask.mean()) if len(valid_mask) > 0 else 0.0
    if valid_mask.sum() < 2:
        return float("nan"), coverage
    return safe_spearman(predictions[valid_mask], targets[valid_mask]), coverage


def evaluate_all_statistical_baselines(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, float]:
    """Evaluate Drug-only, Cell-only, and Additive baselines on a given train/test split."""
    targets = test_df["LN_IC50"]
    sp_drug, cov_drug = compute_valid_spearman(
        compute_drug_only_predictions(train_df, test_df), targets
    )
    sp_cell, cov_cell = compute_valid_spearman(
        compute_cell_only_predictions(train_df, test_df), targets
    )
    sp_add, cov_add = compute_valid_spearman(
        compute_additive_predictions(train_df, test_df), targets
    )
    return {
        "drug_only": sp_drug,
        "drug_cov": cov_drug,
        "cell_only": sp_cell,
        "cell_cov": cov_cell,
        "additive": sp_add,
        "add_cov": cov_add,
    }


def compute_within_cell_spearman(
    preds: np.ndarray,
    trues: np.ndarray,
    cell_ids: np.ndarray,
    min_n: int = 10,
) -> tuple[float, float, int]:
    """Compute mean, std, and count of Spearman correlations evaluated within each cell line."""
    eval_df = pd.DataFrame({"pred": preds, "true": trues, "ModelID": cell_ids})
    stats_df = compute_grouped_regression_stats(eval_df, "ModelID", min_n=min_n)
    valid = stats_df["spearman"].dropna().to_numpy(dtype=np.float64)
    if len(valid) == 0:
        return float("nan"), float("nan"), 0
    return float(np.mean(valid)), float(np.std(valid)), int(len(valid))


def evaluate_nearest_drug_scaffold_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> float:
    """Predict each unseen test drug's response using the mean LN_IC50 of the closest training drug."""
    train_drugs = train_df.groupby("DRUG_NAME", as_index=False).agg(
        {"Morgan_FP": "first", "LN_IC50": "mean"}
    )
    test_drugs = test_df.drop_duplicates(subset=["DRUG_NAME"])[
        ["DRUG_NAME", "Morgan_FP"]
    ].copy()

    similarity_matrix = cosine_similarity(
        np.stack(test_drugs["Morgan_FP"].values),
        np.stack(train_drugs["Morgan_FP"].values),
    )
    nearest_means = train_drugs.iloc[similarity_matrix.argmax(axis=1)]["LN_IC50"].values
    nearest_map = dict(zip(test_drugs["DRUG_NAME"].values, nearest_means))
    return safe_spearman(test_df["DRUG_NAME"].map(nearest_map), test_df["LN_IC50"])
