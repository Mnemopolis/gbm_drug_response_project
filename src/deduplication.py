"""
Data deduplication and multimodal alignment module.

Single Responsibility Principle (SRP):
Isolates all replicate-averaging and key-deduplication logic so replicate
measurements never leak across train and test splits.
"""

import pandas as pd

from src.utils import find_matching_column, normalize_drug_name


def deduplicate_gdsc_drug_cell_pairs(raw_gdsc_gbm: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate replicate (ModelID, DRUG_NAME) dose-response measurements by
    averaging LN_IC50 values to prevent replicate leakage between splits.
    """
    return raw_gdsc_gbm.groupby(["ModelID", "DRUG_NAME"], as_index=False).agg(
        {"LN_IC50": "mean"}
    )


def deduplicate_drug_smiles_table(drug_raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw drug SMILES records, normalize alphanumeric drug keys, and drop
    duplicate drug names and normalized keys.
    """
    name_col = find_matching_column(["drug_name", "DRUG_NAME"], drug_raw_df.columns)
    smiles_col = find_matching_column(
        ["canonical_smiles", "SMILES"], drug_raw_df.columns
    )

    drug_info = (
        drug_raw_df[[name_col, smiles_col]]
        .dropna()
        .drop_duplicates(subset=[name_col])
        .copy()
    )
    drug_info.columns = ["DRUG_NAME", "SMILES"]
    drug_info["DRUG_KEY"] = drug_info["DRUG_NAME"].map(normalize_drug_name)
    drug_info = drug_info.drop_duplicates(subset=["DRUG_KEY"]).reset_index(drop=True)
    return drug_info


def merge_multimodal_dataset(
    gdsc_grouped: pd.DataFrame,
    drug_info: pd.DataFrame,
    expr_gbm: pd.DataFrame,
    mut_gbm: pd.DataFrame,
    mgmt_expr: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    """
    Prefix multi-omics feature columns to prevent gene symbol collisions and merge
    drug response labels, molecular features, and cell line omics into a unified table.
    """
    expr_prefixed = expr_gbm.add_prefix("EXPR_")
    mut_prefixed = mut_gbm.add_prefix("MUT_")
    mgmt_prefixed = mgmt_expr.copy()

    expr_cols = list(expr_prefixed.columns)
    mut_cols = list(mut_prefixed.columns)
    mgmt_cols = list(mgmt_prefixed.columns)

    gdsc_keyed = gdsc_grouped.copy()
    gdsc_keyed["DRUG_KEY"] = gdsc_keyed["DRUG_NAME"].map(normalize_drug_name)

    drug_cols = [
        c
        for c in ["DRUG_KEY", "SMILES", "Morgan_FP", "Scaffold"]
        if c in drug_info.columns
    ]
    merged_df = gdsc_keyed.merge(drug_info[drug_cols], on="DRUG_KEY", how="inner")
    merged_df = merged_df.merge(
        expr_prefixed, left_on="ModelID", right_index=True, how="inner"
    )
    merged_df = merged_df.merge(
        mut_prefixed, left_on="ModelID", right_index=True, how="inner"
    )
    merged_df = merged_df.merge(
        mgmt_prefixed, left_on="ModelID", right_index=True, how="left"
    )

    return merged_df, expr_cols, mut_cols, mgmt_cols
