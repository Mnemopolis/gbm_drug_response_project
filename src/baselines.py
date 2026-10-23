"""
Non-deep-learning statistical and chemical similarity baselines for diagnosing
shortcut learning and evaluating OOD generalization across splits.
"""

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
    """
    Predict test LN_IC50 using an additive main-effects model (no drug-cell interaction):
    pred = drug_mean + cell_mean - global_train_mean.
    """
    global_mean = float(train_df["LN_IC50"].mean())
    pred_drug = compute_drug_only_predictions(train_df, test_df)
    pred_cell = compute_cell_only_predictions(train_df, test_df)
    return pred_drug + pred_cell - global_mean


def compute_valid_spearman(
    predictions: pd.Series, targets: pd.Series
) -> tuple[float, float]:
    """Compute Spearman rank correlation and valid coverage fraction for non-NaN predictions."""
    valid_mask = predictions.notna() & targets.notna()
    coverage = float(valid_mask.mean()) if len(valid_mask) > 0 else 0.0
    if valid_mask.sum() < 2:
        return float("nan"), coverage
    corr = float(spearmanr(predictions[valid_mask], targets[valid_mask]).correlation)
    return corr, coverage


def evaluate_all_statistical_baselines(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, float]:
    """Evaluate Drug-only, Cell-only, and Additive baselines on a given train/test split."""
    targets = test_df["LN_IC50"]
    pred_drug = compute_drug_only_predictions(train_df, test_df)
    pred_cell = compute_cell_only_predictions(train_df, test_df)
    pred_add = compute_additive_predictions(train_df, test_df)

    sp_drug, cov_drug = compute_valid_spearman(pred_drug, targets)
    sp_cell, cov_cell = compute_valid_spearman(pred_cell, targets)
    sp_add, cov_add = compute_valid_spearman(pred_add, targets)

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
    """
    Compute the mean and standard deviation of Spearman correlations calculated
    independently within each cell line (requires at least `min_n` drugs per cell line).
    """
    eval_df = pd.DataFrame({"pred": preds, "true": trues, "cell": cell_ids})
    correlations = []
    for _, group in eval_df.groupby("cell"):
        if len(group) < min_n:
            continue
        if group["true"].std() < 1e-8 or group["pred"].std() < 1e-8:
            continue
        sp = float(spearmanr(group["pred"], group["true"]).correlation)
        if not np.isnan(sp):
            correlations.append(sp)

    if not correlations:
        return float("nan"), float("nan"), 0
    return float(np.mean(correlations)), float(np.std(correlations)), len(correlations)


def evaluate_nearest_drug_scaffold_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> float:
    """
    For chemical OOD (scaffold split), predict each unseen test drug's response using the
    mean LN_IC50 of the most structurally similar training drug (via Morgan FP cosine similarity).
    """
    train_drugs = train_df.groupby("DRUG_NAME", as_index=False).agg(
        {"Morgan_FP": "first", "LN_IC50": "mean"}
    )
    test_drugs = test_df.drop_duplicates(subset=["DRUG_NAME"])[
        ["DRUG_NAME", "Morgan_FP"]
    ].copy()

    train_fps = np.stack(train_drugs["Morgan_FP"].values)
    test_fps = np.stack(test_drugs["Morgan_FP"].values)

    similarity_matrix = cosine_similarity(test_fps, train_fps)
    nearest_train_indices = similarity_matrix.argmax(axis=1)
    nearest_drug_means = train_drugs.iloc[nearest_train_indices]["LN_IC50"].values

    nearest_mean_map = dict(zip(test_drugs["DRUG_NAME"].values, nearest_drug_means))
    predicted_nn = test_df["DRUG_NAME"].map(nearest_mean_map)

    return float(spearmanr(predicted_nn, test_df["LN_IC50"]).correlation)
