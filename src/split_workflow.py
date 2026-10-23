"""
Stage 1 Workflow Orchestrator: Data Acquisition -> Deduplication -> Fingerprints
-> Splitting -> Leakage-Free Standardization/PCA -> Pipeline Audit -> Persistence.
"""

import pandas as pd

from src.acquisition import (
    convert_and_extract_methylation_features,
    extract_expression_features,
    extract_mutation_features,
    load_raw_gdsc_gbm_records,
)
from src.config import (
    EXPR_FILE,
    GDSC_FILE,
    METHYL_CSV_FILE,
    METHYL_TXT_FILE,
    MODEL_FILE,
    MUT_FILE,
    SMILES_FILE,
    SPLIT_NAMES,
    get_split_data_paths,
)
from src.deduplication import deduplicate_gdsc_drug_cell_pairs, merge_multimodal_dataset
from src.eda import compute_cohort_eda_summary, plot_label_distribution_across_splits
from src.fingerprints import build_drug_feature_table
from src.pipeline_audit import run_full_pipeline_audit
from src.splitting import partition_dataframe_by_split
from src.standardization import fit_transform_cell_pca_features


def run_split_workflow(
    save_splits: bool = True,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Execute the complete Stage 1 data preprocessing and leakage-free splitting workflow:
    1. Acquire raw GDSC2 GBM records and deduplicate drug-cell pairs
    2. Extract expression, somatic mutation, and RRBS methylation features
    3. Generate 2048-bit Morgan fingerprints and Bemis-Murcko scaffolds
    4. Merge multimodal dataset and compute EDA summary
    5. Partition into Random, Leave-Cell-Out, and Scaffold splits with train-only PCA
    6. Audit all splits via `pipeline_audit` and save `.pkl` artifacts to `Models/`
    """
    print("=" * 70)
    print("WORKFLOW 1: Multi-Omics Acquisition, Deduplication & Leakage-Free Splitting")
    print("=" * 70)

    raw_gbm = load_raw_gdsc_gbm_records(GDSC_FILE, MODEL_FILE)
    gdsc_grouped = deduplicate_gdsc_drug_cell_pairs(raw_gbm)
    print(
        f"[Deduplication] {len(gdsc_grouped)} unique drug-cell pairs "
        f"(Cells={gdsc_grouped['ModelID'].nunique()}, Drugs={gdsc_grouped['DRUG_NAME'].nunique()})"
    )

    valid_cells = gdsc_grouped["ModelID"].unique()
    expr_gbm, mgmt_expr = extract_expression_features(EXPR_FILE, valid_cells)
    mut_gbm = extract_mutation_features(MUT_FILE, valid_cells)
    print(
        f"[Acquisition] Expression={expr_gbm.shape} | Somatic Mutations={mut_gbm.shape}"
    )

    if METHYL_TXT_FILE.exists():
        mgmt_methyl = convert_and_extract_methylation_features(
            METHYL_TXT_FILE, METHYL_CSV_FILE, MODEL_FILE
        )
        if mgmt_methyl is not None:
            print(
                f"[Methylation] Converted RRBS table & extracted {len(mgmt_methyl)} MGMT records."
            )

    drug_info = build_drug_feature_table(SMILES_FILE)
    print(
        f"[Fingerprints] Featurized {len(drug_info)} unique drugs with Morgan FP & Murcko Scaffolds."
    )

    final_df, expr_cols, mut_cols, mgmt_cols = merge_multimodal_dataset(
        gdsc_grouped, drug_info, expr_gbm, mut_gbm, mgmt_expr
    )
    eda_summary = compute_cohort_eda_summary(final_df)
    print(
        f"[EDA Summary] Merged records={eda_summary['n_records']} | "
        f"Cells={eda_summary['n_unique_cells']} | Drugs={eda_summary['n_unique_drugs']} | "
        f"Scaffolds={eda_summary['n_unique_scaffolds']} | "
        f"LN_IC50={eda_summary['ic50_mean']:.2f}+/-{eda_summary['ic50_std']:.2f}"
    )

    splits_dict: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for split_name in SPLIT_NAMES:
        raw_train, raw_test = partition_dataframe_by_split(final_df, split_name)
        train_final, test_final = fit_transform_cell_pca_features(
            raw_train, raw_test, expr_cols, mut_cols, mgmt_cols
        )
        splits_dict[split_name] = (train_final, test_final)
        print(
            f"  [{split_name:<10s}] train={train_final.shape} | test={test_final.shape}"
        )

        if save_splits:
            paths = get_split_data_paths(split_name)
            train_final.to_pickle(paths["train"])
            test_final.to_pickle(paths["test"])

    audit_df = run_full_pipeline_audit(splits_dict)
    print("\n[Pipeline Audit] Anti-Leakage Verification Passed:")
    print(audit_df.to_string(index=False))

    if save_splits:
        fig_path = plot_label_distribution_across_splits(splits_dict)
        print(f"[EDA Plot] Saved split distribution histogram to {fig_path}")

    return splits_dict
