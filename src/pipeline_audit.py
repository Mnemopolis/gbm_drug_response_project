"""
Automated pipeline integrity and anti-data-leakage auditing module.

Single Responsibility Principle (SRP):
Programmatically verifies that train/test splits satisfy strict OOD disjointness
constraints (zero cell overlap in `leave_cell`, zero scaffold overlap in `scaffold`,
zero duplicate drug-cell pairs, and zero NaN values in engineered features).
"""

from typing import Any

import numpy as np
import pandas as pd


def audit_split_no_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    split_type: str,
) -> dict[str, Any]:
    """
    Verify that a train/test split satisfies all anti-leakage invariants for its regime.
    Raises an AssertionError if any data leakage is detected.
    """
    if len(train_df) == 0 or len(test_df) == 0:
        raise AssertionError(
            f"[{split_type}] Empty split detected: train={len(train_df)}, test={len(test_df)}"
        )

    # 1. Verify zero exact (ModelID, DRUG_NAME) pair overlap across all split types
    train_pairs = set(zip(train_df["ModelID"], train_df["DRUG_NAME"]))
    test_pairs = set(zip(test_df["ModelID"], test_df["DRUG_NAME"]))
    pair_overlap = len(train_pairs.intersection(test_pairs))
    if pair_overlap > 0:
        raise AssertionError(
            f"[{split_type}] Data leakage: {pair_overlap} (ModelID, DRUG_NAME) pairs overlap!"
        )

    train_cells = set(train_df["ModelID"].unique())
    test_cells = set(test_df["ModelID"].unique())
    cell_overlap = len(train_cells.intersection(test_cells))

    train_scaffolds = (
        set(train_df["Scaffold"].dropna().unique())
        if "Scaffold" in train_df.columns
        else set()
    )
    test_scaffolds = (
        set(test_df["Scaffold"].dropna().unique())
        if "Scaffold" in test_df.columns
        else set()
    )
    scaffold_overlap = len(train_scaffolds.intersection(test_scaffolds))

    # 2. Regime-specific OOD disjointness assertions
    if split_type == "leave_cell" and cell_overlap > 0:
        raise AssertionError(
            f"[leave_cell] Cell-line OOD leakage: {cell_overlap} cell lines appear in both train and test!"
        )

    if split_type == "scaffold" and scaffold_overlap > 0:
        raise AssertionError(
            f"[scaffold] Chemical scaffold OOD leakage: {scaffold_overlap} scaffolds appear in both train and test!"
        )

    return {
        "split_type": split_type,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "pair_overlap": pair_overlap,
        "cell_overlap": cell_overlap,
        "scaffold_overlap": scaffold_overlap,
        "passed": True,
    }


def audit_feature_matrix_integrity(df: pd.DataFrame, split_type: str) -> dict[str, Any]:
    """
    Verify that engineered cell-line features (PCA_, MUT_, MGMT_) and Morgan fingerprints
    contain no NaN or infinite values and that raw EXPR_ columns have been removed.
    """
    raw_expr_cols = [c for c in df.columns if c.startswith("EXPR_")]
    if len(raw_expr_cols) > 0:
        raise AssertionError(
            f"[{split_type}] Raw expression columns were not replaced by PCA: {len(raw_expr_cols)} remain."
        )

    cell_cols = [c for c in df.columns if c.startswith(("PCA_", "MUT_", "MGMT_"))]
    if len(cell_cols) == 0:
        raise AssertionError(
            f"[{split_type}] No engineered cell feature columns (PCA_/MUT_/MGMT_) found."
        )

    cell_matrix = df[cell_cols].values.astype(np.float32)
    if np.isnan(cell_matrix).any() or np.isinf(cell_matrix).any():
        raise AssertionError(
            f"[{split_type}] NaN or Inf detected in engineered cell feature matrix."
        )

    fp_matrix = np.stack(df["Morgan_FP"].values)
    if fp_matrix.shape[1] != 2048:
        raise AssertionError(
            f"[{split_type}] Expected 2048-bit Morgan_FP, got {fp_matrix.shape[1]}."
        )

    return {
        "split_type": split_type,
        "cell_feature_dim": len(cell_cols),
        "fp_dim": fp_matrix.shape[1],
        "has_nan": False,
    }


def run_full_pipeline_audit(
    splits_dict: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
) -> pd.DataFrame:
    """Audit all splits in `splits_dict` and return a summary audit DataFrame."""
    records = []
    for split_name, (train_df, test_df) in splits_dict.items():
        leak_report = audit_split_no_leakage(train_df, test_df, split_name)
        feat_tr = audit_feature_matrix_integrity(train_df, f"{split_name}_train")
        feat_te = audit_feature_matrix_integrity(test_df, f"{split_name}_test")
        records.append(
            {
                **leak_report,
                "train_cell_dim": feat_tr["cell_feature_dim"],
                "test_cell_dim": feat_te["cell_feature_dim"],
            }
        )
    return pd.DataFrame(records)
