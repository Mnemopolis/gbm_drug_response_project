"""
Small, reproducible GBM multi-omics & drug inputs for the Lectrace presentation.
Prefixed with '_' so the Lectrace CLI steps over helper data loading automatically.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GBMCellLineSample:
    model_id: str
    cell_line_name: str
    oncotree_subtype: str
    mgmt_expr: float
    mutations: dict[str, int]
    raw_expr_vector: list[float]
    pca_omics_vector: list[float]

    def __lectrace__(self) -> dict:
        return {
            "model_id": self.model_id,
            "name": self.cell_line_name,
            "subtype": self.oncotree_subtype,
            "MGMT_Expr": round(self.mgmt_expr, 3),
            "hotspot_mutations": self.mutations,
            "pca_4d": [round(v, 3) for v in self.pca_omics_vector],
        }


@dataclass(frozen=True)
class GBMDrugSample:
    drug_name: str
    smiles: str
    murcko_scaffold: str
    split_assignment: str
    max_tanimoto_to_train: float
    morgan_fp_8d: list[float]
    true_mean_ln_ic50: float
    pred_mean_ln_ic50_random: float
    pred_mean_ln_ic50_scaffold: float

    def __lectrace__(self) -> dict:
        return {
            "drug_name": self.drug_name,
            "scaffold": self.murcko_scaffold,
            "split_role": self.split_assignment,
            "max_train_sim": round(self.max_tanimoto_to_train, 3),
            "morgan_fp_8d": [int(b) for b in self.morgan_fp_8d],
            "true_LN_IC50": round(self.true_mean_ln_ic50, 2),
            "pred_scaffold_LN_IC50": round(self.pred_mean_ln_ic50_scaffold, 2),
        }


COHORT_SUMMARY = {
    "raw_gdsc_gbm_records": 8603,
    "deduplicated_pairs": 5107,
    "unique_cell_lines": 24,
    "unique_drugs": 243,
    "raw_expr_genes": 19138,
    "active_hotspot_mutations": 20,
    "mgmt_features": 1,
    "morgan_fp_bits": 2048,
    "morgan_radius": 2,
    "cell_feat_dim_random_scaffold": 44,
    "cell_feat_dim_leave_cell": 40,
}

SAMPLE_CELL_LINES: list[GBMCellLineSample] = [
    GBMCellLineSample(
        model_id="ACH-000075",
        cell_line_name="U-87 MG",
        oncotree_subtype="Glioblastoma (GB)",
        mgmt_expr=0.420,
        mutations={"PTEN": 1, "TP53": 0, "EGFR": 0, "RB1": 0},
        raw_expr_vector=[6.12, 4.85, 7.31, 2.14, 5.60, 3.92],
        pca_omics_vector=[1.42, -0.88, 0.35, 0.19],
    ),
    GBMCellLineSample(
        model_id="ACH-000571",
        cell_line_name="T98G",
        oncotree_subtype="Glioblastoma (GB)",
        mgmt_expr=5.840,
        mutations={"PTEN": 1, "TP53": 1, "EGFR": 0, "RB1": 0},
        raw_expr_vector=[5.90, 6.42, 6.80, 4.95, 5.10, 4.45],
        pca_omics_vector=[-1.15, 1.24, -0.42, 0.61],
    ),
    GBMCellLineSample(
        model_id="ACH-000098",
        cell_line_name="GAMG",
        oncotree_subtype="Glioblastoma (GB)",
        mgmt_expr=1.950,
        mutations={"PTEN": 0, "TP53": 1, "EGFR": 1, "RB1": 0},
        raw_expr_vector=[7.05, 5.18, 6.12, 3.40, 6.25, 3.15],
        pca_omics_vector=[0.64, 0.52, 1.18, -0.47],
    ),
    GBMCellLineSample(
        model_id="ACH-000595",
        cell_line_name="LN-229",
        oncotree_subtype="Glioblastoma (GB)",
        mgmt_expr=0.780,
        mutations={"PTEN": 0, "TP53": 1, "EGFR": 0, "RB1": 1},
        raw_expr_vector=[6.48, 4.30, 7.10, 2.85, 5.40, 3.70],
        pca_omics_vector=[0.91, -0.45, -0.76, -0.33],
    ),
]

SAMPLE_DRUGS: list[GBMDrugSample] = [
    GBMDrugSample(
        drug_name="Temozolomide",
        smiles="CN1C(=O)N2C=NC(=C2N=N1)C(=O)N",
        murcko_scaffold="O=c1ncc2ncn([H])c2n1",
        split_assignment="test_scaffold",
        max_tanimoto_to_train=0.284,
        morgan_fp_8d=[1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0],
        true_mean_ln_ic50=6.45,
        pred_mean_ln_ic50_random=6.17,
        pred_mean_ln_ic50_scaffold=3.32,
    ),
    GBMDrugSample(
        drug_name="Irinotecan",
        smiles="CCC1=C2CN3C(=CC4=C(C3=O)COC(=O)C4(CC)O)C2=NC5=C1C=C(C=C5)OC(=O)N6CCC(CC6)N7CCCCC7",
        murcko_scaffold="O=c1cc2c3nc4ccccc4c3cn2c2c1COC(=O)C2",
        split_assignment="test_scaffold",
        max_tanimoto_to_train=0.241,
        morgan_fp_8d=[0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0],
        true_mean_ln_ic50=2.764,
        pred_mean_ln_ic50_random=2.69,
        pred_mean_ln_ic50_scaffold=-1.014,
    ),
    GBMDrugSample(
        drug_name="Dabrafenib",
        smiles="CC(C)(C)C1=NC(=C(S1)C2=NC(=NC=C2)N)C3=C(C(=CC=C3)NS(=O)(=O)C4=C(C=CC=C4F)F)F",
        murcko_scaffold="c1ccc(S(=O)(=O)Nc2cccc(-c3cncnc3-c3cscn3)c2)cc1",
        split_assignment="train_scaffold",
        max_tanimoto_to_train=1.000,
        morgan_fp_8d=[1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
        true_mean_ln_ic50=3.12,
        pred_mean_ln_ic50_random=3.18,
        pred_mean_ln_ic50_scaffold=3.15,
    ),
    GBMDrugSample(
        drug_name="Docetaxel",
        smiles="CC1=C2C(C(=O)C3(C(CC4C(C3C(C(C2(C)C)(CC1OC(=O)C(C(C5=CC=CC=C5)NC(=O)OC(C)(C)C)O)O)OC(=O)C6=CC=CC=C6)(CO4)OC(=O)C)O)C)O",
        murcko_scaffold="O=C(OC1CC2OC23CC2C(OC(=O)c4ccccc4)C1C1(O)CC(OC(=O)CCc4ccccc4)C=C1C32)c1ccccc1",
        split_assignment="train_scaffold",
        max_tanimoto_to_train=1.000,
        morgan_fp_8d=[0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        true_mean_ln_ic50=-4.12,
        pred_mean_ln_ic50_random=-3.95,
        pred_mean_ln_ic50_scaffold=-3.88,
    ),
]

DEMO_MLP_WEIGHTS = {
    "y_mean": 2.850,
    "y_std": 2.580,
    "W_drug_1": [
        [0.35, -0.20, 0.45, 0.15, -0.30, 0.25, 0.10, -0.15],
        [-0.10, 0.50, 0.20, -0.25, 0.40, 0.30, -0.20, 0.35],
        [0.40, 0.15, -0.35, 0.45, 0.10, -0.25, 0.30, 0.20],
        [0.25, -0.30, 0.30, 0.35, -0.15, 0.40, -0.10, 0.15],
    ],
    "b_drug_1": [0.10, -0.05, 0.08, 0.12],
    "W_cell_1": [
        [0.42, -0.38, 0.25, 0.18, 0.30, -0.45],
        [-0.22, 0.35, 0.40, -0.15, 0.20, 0.52],
        [0.31, 0.28, -0.20, 0.36, -0.12, 0.15],
        [0.19, -0.14, 0.33, 0.22, 0.41, -0.28],
    ],
    "b_cell_1": [0.05, 0.10, -0.04, 0.06],
    "w_head_random": [0.62, -0.58, 0.48, 0.54, -0.35, 0.68, 0.29, -0.31],
    "b_head_random": 0.42,
    "w_head_scaffold": [0.08, -0.05, 0.06, 0.04, -0.22, 0.38, 0.15, -0.18],
    "b_head_scaffold": 0.05,
}

BENCHMARK_RESULTS = {
    "random": {
        "split_label": "Random (In-Distribution)",
        "train_pairs": 3472,
        "val_pairs": 613,
        "test_pairs": 1022,
        "train_cells": 24,
        "test_cells": 24,
        "train_drugs": 243,
        "test_drugs": 236,
        "cell_only_spearman": 0.2107,
        "drug_only_spearman": 0.8349,
        "additive_spearman": 0.8752,
        "nearest_drug_spearman": None,
        "mlp_global_spearman_mean": 0.8744,
        "mlp_global_spearman_std": 0.0051,
        "mlp_global_rmse_mean": 1.0346,
        "mlp_global_rmse_std": 0.0211,
        "mlp_within_cell_spearman_mean": 0.8541,
        "mlp_within_cell_spearman_std": 0.0576,
        "drug_only_within_cell_spearman_mean": 0.8554,
        "drug_only_within_cell_spearman_std": 0.0598,
        "per_drug_median_spearman": 0.500,
        "per_drug_median_rmse": 0.893,
        "pred_std": 2.267,
        "true_std": 2.631,
        "shrinkage_ratio": 0.862,
        "mean_bias_err": -0.055,
        "large_error_rate_gt_2log": 0.060,
    },
    "leave_cell": {
        "split_label": "Leave-Cell-Out (Biological OOD)",
        "train_pairs": 3611,
        "val_pairs": 638,
        "test_pairs": 858,
        "train_cells": 20,
        "test_cells": 4,
        "train_drugs": 243,
        "test_drugs": 243,
        "cell_only_spearman": None,
        "drug_only_spearman": 0.8362,
        "additive_spearman": None,
        "nearest_drug_spearman": None,
        "mlp_global_spearman_mean": 0.6947,
        "mlp_global_spearman_std": 0.2979,
        "mlp_stable_seeds_spearman_mean": 0.8436,
        "mlp_global_rmse_mean": 1.3970,
        "mlp_global_rmse_std": 0.6115,
        "mlp_within_cell_spearman_mean": 0.8460,
        "mlp_within_cell_spearman_std": 0.0608,
        "drug_only_within_cell_spearman_mean": 0.8449,
        "drug_only_within_cell_spearman_std": 0.0602,
        "per_drug_median_spearman": 0.500,
        "per_drug_median_rmse": 0.887,
        "pred_std": 2.209,
        "true_std": 2.592,
        "shrinkage_ratio": 0.852,
        "mean_bias_err": -0.147,
        "large_error_rate_gt_2log": 0.070,
        "per_seed_spearman": [0.8462, 0.8388, 0.0989, 0.8476, 0.8419],
    },
    "scaffold": {
        "split_label": "Bemis-Murcko Scaffold (Chemical OOD)",
        "train_pairs": 3447,
        "val_pairs": 609,
        "test_pairs": 1051,
        "train_cells": 24,
        "test_cells": 24,
        "train_drugs": 196,
        "test_drugs": 47,
        "cell_only_spearman": 0.2494,
        "drug_only_spearman": None,
        "additive_spearman": None,
        "nearest_drug_spearman": 0.1445,
        "mlp_global_spearman_mean": 0.2905,
        "mlp_global_spearman_std": 0.0045,
        "mlp_global_rmse_mean": 2.1951,
        "mlp_global_rmse_std": 0.0081,
        "mlp_within_cell_spearman_mean": 0.1735,
        "mlp_within_cell_spearman_std": 0.0813,
        "drug_only_within_cell_spearman_mean": None,
        "drug_only_within_cell_spearman_std": None,
        "per_drug_median_spearman": 0.530,
        "per_drug_median_rmse": 1.644,
        "pred_std": 1.630,
        "true_std": 2.459,
        "shrinkage_ratio": 0.663,
        "mean_bias_err": -0.316,
        "large_error_rate_gt_2log": 0.368,
    },
}

SCAFFOLD_NOVELTY_BINS = [
    {
        "similarity_bin": "< 0.30 (High Novelty)",
        "drug_count": 15,
        "mean_rmse": 2.112,
        "median_rmse": 2.161,
    },
    {
        "similarity_bin": "0.30 - 0.50 (Moderate)",
        "drug_count": 23,
        "mean_rmse": 2.120,
        "median_rmse": 1.871,
    },
    {
        "similarity_bin": "0.50 - 0.70 (Near-Analog)",
        "drug_count": 4,
        "mean_rmse": 2.020,
        "median_rmse": 1.463,
    },
    {
        "similarity_bin": "> 0.70 (Close Scaffold)",
        "drug_count": 2,
        "mean_rmse": 1.253,
        "median_rmse": 1.253,
    },
]
