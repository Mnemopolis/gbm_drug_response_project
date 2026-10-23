"""
Exploratory Data Analysis (EDA) and split label distribution visualization module.

Single Responsibility Principle (SRP):
Computes summary statistics of the multi-omics cohort (cell counts, drug counts,
scaffold counts, MGMT missingness, LN_IC50 distribution) and plots split histograms.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import RESULTS_DIR


def compute_cohort_eda_summary(df: pd.DataFrame) -> dict[str, any]:
    """Compute high-level exploratory statistics for the merged GBM drug-response dataset."""
    mgmt_missing_rate = (
        float(df["MGMT_Expr"].isna().mean()) if "MGMT_Expr" in df.columns else 0.0
    )
    return {
        "n_records": len(df),
        "n_unique_cells": int(df["ModelID"].nunique()),
        "n_unique_drugs": int(df["DRUG_NAME"].nunique()),
        "n_unique_scaffolds": int(df["Scaffold"].dropna().nunique())
        if "Scaffold" in df.columns
        else 0,
        "ic50_mean": float(df["LN_IC50"].mean()),
        "ic50_std": float(df["LN_IC50"].std()),
        "ic50_min": float(df["LN_IC50"].min()),
        "ic50_max": float(df["LN_IC50"].max()),
        "mgmt_missing_rate": mgmt_missing_rate,
    }


def plot_label_distribution_across_splits(
    splits_dict: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    output_path: Path = RESULTS_DIR / "three_split.png",
) -> Path:
    """Plot train vs. test LN_IC50 histograms across Random, Leave-Cell, and Scaffold splits."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    display_titles = [
        ("random", "Random Split"),
        ("leave_cell", "Leave-Cell Split"),
        ("scaffold", "Scaffold Split"),
    ]

    for ax, (split_key, title) in zip(axes, display_titles):
        train_df, test_df = splits_dict[split_key]
        ax.hist(train_df["LN_IC50"], bins=30, alpha=0.5, label="Train")
        ax.hist(test_df["LN_IC50"], bins=30, alpha=0.5, label="Test")
        ax.set_title(title)
        ax.set_xlabel("LN_IC50")
        ax.set_ylabel("Count")
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
