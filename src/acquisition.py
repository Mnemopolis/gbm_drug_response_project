"""
Data acquisition and raw multi-omics ingestion module.

Single Responsibility Principle (SRP):
Handles reading raw GDSC2 dose-response CSVs, DepMap cell-line metadata,
gene expression matrices, somatic mutation matrices, and RRBS methylation tables.
Does not perform deduplication, splitting, or PCA transformation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    EXPR_FILE,
    GDSC_FILE,
    METHYL_CSV_FILE,
    METHYL_TXT_FILE,
    MODEL_FILE,
    MUT_FILE,
)


def load_depmap_model_mapping(model_file: Path = MODEL_FILE) -> pd.DataFrame:
    """Load DepMap cell line metadata and return valid ModelID-to-SangerModelID mappings."""
    model_df = pd.read_csv(model_file)
    cols = [c for c in ["ModelID", "SangerModelID", "CellLineName"] if c in model_df.columns]
    return model_df[cols].dropna(subset=["ModelID", "SangerModelID"]).copy()


def load_raw_gdsc_gbm_records(
    gdsc_file: Path = GDSC_FILE,
    model_file: Path = MODEL_FILE,
) -> pd.DataFrame:
    """Load GDSC2 records, filter for Glioblastoma (GBM), and map Sanger IDs to DepMap ModelID."""
    gdsc_df = pd.read_csv(gdsc_file, low_memory=False)
    gdsc_gbm = gdsc_df[
        gdsc_df["CANCER_TYPE"].str.contains("Glioblastoma", case=False, na=False)
    ].copy()

    model_mapping = load_depmap_model_mapping(model_file)[["ModelID", "SangerModelID"]]
    merged_gbm = pd.merge(
        gdsc_gbm,
        model_mapping,
        left_on="SANGER_MODEL_ID",
        right_on="SangerModelID",
        how="inner",
    )
    return merged_gbm


def extract_expression_features(
    expr_file: Path = EXPR_FILE,
    valid_cells: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load protein-coding gene expression matrix (TPM log+1) for valid GBM cell lines
    and extract the MGMT expression proxy feature separately.
    """
    expr_df = pd.read_csv(expr_file, index_col=0, low_memory=False)
    if valid_cells is not None:
        expr_gbm = expr_df.loc[expr_df.index.intersection(valid_cells)].copy()
    else:
        expr_gbm = expr_df.copy()

    mgmt_cols = [col for col in expr_df.columns if "MGMT" in col]
    if not mgmt_cols:
        raise ValueError("No MGMT column found in gene expression matrix.")
    mgmt_col = mgmt_cols[0]
    mgmt_expr_df = expr_gbm[[mgmt_col]].rename(columns={mgmt_col: "MGMT_Expr"})

    return expr_gbm, mgmt_expr_df


def extract_mutation_features(
    mut_file: Path = MUT_FILE,
    valid_cells: np.ndarray | None = None,
) -> pd.DataFrame:
    """
    Load somatic hotspot mutation matrix for valid GBM cell lines, keeping default model
    entries and filtering out genes with zero mutations across the cohort.
    """
    mut_df = pd.read_csv(mut_file, low_memory=False)
    if "IsDefaultEntryForModel" in mut_df.columns:
        mut_df = mut_df[mut_df["IsDefaultEntryForModel"] == "Yes"].copy()

    mut_df = mut_df.set_index("ModelID")
    metadata_cols = [
        "SequencingID",
        "ModelConditionID",
        "IsDefaultEntryForModel",
        "IsDefaultEntryForMC",
    ]
    cols_to_drop = [col for col in metadata_cols if col in mut_df.columns]
    mut_df = mut_df.drop(columns=cols_to_drop)

    if valid_cells is not None:
        mut_gbm = mut_df.loc[mut_df.index.intersection(valid_cells)].copy()
    else:
        mut_gbm = mut_df.copy()

    non_zero_mask = (mut_gbm != 0).any(axis=0)
    return mut_gbm.loc[:, non_zero_mask]


def convert_and_extract_methylation_features(
    methyl_txt_file: Path = METHYL_TXT_FILE,
    methyl_csv_file: Path = METHYL_CSV_FILE,
    model_file: Path = MODEL_FILE,
) -> pd.DataFrame | None:
    """
    Convert raw CCLE RRBS TSS 1kb tab-delimited file into CSV format and extract
    MGMT promoter methylation mapped to DepMap ModelID.
    """
    if not methyl_txt_file.exists():
        return None

    df_methylation = pd.read_csv(methyl_txt_file, sep="\t", index_col=0, low_memory=False)
    df_methylation.to_csv(methyl_csv_file)

    mgmt_rows = df_methylation.loc[
        df_methylation.index.str.contains("MGMT", case=False, na=False)
    ]
    mgmt_methyl = mgmt_rows.T
    if mgmt_methyl.shape[1] >= 1:
        mgmt_methyl = mgmt_methyl.iloc[:, [0]].copy()
        mgmt_methyl.columns = ["MGMT_Methylation"]
    else:
        return None

    model_df = pd.read_csv(model_file)
    name_to_id = dict(zip(model_df["CellLineName"], model_df["ModelID"]))
    mgmt_methyl.index = mgmt_methyl.index.map(name_to_id)
    mgmt_methyl = mgmt_methyl[mgmt_methyl.index.notna()].copy()
    return mgmt_methyl
