"""
10-Minute Execution-Explanation Presentation Using Lectrace.
Run locally via the official Lectrace CLI:
    lectrace serve Lectrace/01_gbm_ood_presentation.py
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import math
import sys

_LECTRACE_DIR = Path(__file__).resolve().parent
if str(_LECTRACE_DIR) not in sys.path:
    sys.path.insert(0, str(_LECTRACE_DIR))

try:
    from lectrace import Reference, link, note, plot, table, text
except ImportError:  # Lightweight stub matching official Lectrace API signatures
    @dataclass(frozen=True)
    class Reference:
        title: Optional[str] = None
        authors: Optional[list[str]] = None
        organization: Optional[str] = None
        date: Optional[str] = None
        url: Optional[str] = None
        description: Optional[str] = None
        notes: Optional[str] = None

    def text(message: str, verbatim: bool = False, style: Optional[dict] = None) -> None:
        return None

    def note(message: str) -> None:
        return None

    def table(rows, headers=None, caption=None, style=None) -> None:
        return None

    def plot(spec) -> None:
        return None

    def link(
        arg=None,
        *,
        title=None,
        url=None,
        authors=None,
        date=None,
        description=None,
        organization=None,
        notes=None,
        style=None,
    ) -> None:
        return None

from _project_data import (
    BENCHMARK_RESULTS,
    COHORT_SUMMARY,
    DEMO_MLP_WEIGHTS,
    SAMPLE_CELL_LINES,
    SAMPLE_DRUGS,
    SCAFFOLD_NOVELTY_BINS,
)


def main():
    # -------------------------------------------------------------------------
    # Intro (Minutes 0:00 - 2:30): Healthcare Problem, Motivation & Approach
    # -------------------------------------------------------------------------
    text(r"""
        # Deep Learning for Glioblastoma (GBM) Drug Response & OOD Generalization
        **10-Minute Execution-Explanation Presentation**

        ### 1. Clinical Problem & Motivation (Minutes 0:00 – 2:30)
        - **Clinical Need**: Glioblastoma (GBM) is the most aggressive primary brain tumor ($<15$ months median survival). Predicting drug response ($\ln(\text{IC}_{50})$) from multi-omics profiles is key to precision oncology.
        - **The OOD Challenge**: Models evaluated on **in-distribution random splits** often achieve high correlation ($r_s > 0.85$) via **shortcut learning** (memorizing average drug toxicity) but fail on **unseen chemical scaffolds**.
    """)
    note("""
        [0:00 - 1:30] Introduce the clinical motivation: Does a 0.87 Spearman on random splits mean
        the neural network learned cancer biology (such as MGMT-mediated Temozolomide resistance),
        or is it exploiting a drug-mean shortcut?
    """)

    gdsc_ref = Reference(
        title="GDSC: Genomics of Drug Sensitivity in Cancer",
        authors=["Yang", "Soares", "Greninger", "Garnett"],
        date="2013",
        url="https://www.cancerrxgene.org/",
        description="GDSC2 dose-response dataset providing fitted LN_IC50 values.",
    )
    link(gdsc_ref)
    link(
        title="DepMap: The Cancer Dependency Map",
        authors=["Tsherniak", "Vazquez", "Montgomery", "Golub"],
        date="2017",
        url="https://depmap.org/portal/",
        description="DepMap multi-omics resource (RNA-seq TPM, hotspot mutations, methylation).",
    )

    cohort_meta = COHORT_SUMMARY  # @inspect cohort_meta
    text(r"""
        ### 2. Multimodal Cohort & Two-Tower Architecture
        - **Aligned Cohort**: $5,107$ deduplicated `(drug, cell_line)` pairs across **$24$ GBM cell lines** and **$243$ drugs**.
        - **Two-Tower `BaselineMLP`** (`src/mlp_model.py`): Encodes $2048$-bit Morgan fingerprints ($\mathbf{x}_{\text{drug}}$) and $44$-D multi-omics vectors ($\mathbf{x}_{\text{cell}}$: PCA expression + hotspot mutations + `MGMT`) into $\mathbb{R}^{256}$ embeddings before predicting standardized $\text{LN\_IC50}$.
    """)
    note("""
        [1:30 - 2:30] Point out `cohort_meta` in the Variable Panel, then transition into the
        4 selected execution states.
    """)

    # -------------------------------------------------------------------------
    # Execution State 1 (Minutes 2:30 - 4:30): Leakage-Free OOD Splitting
    # -------------------------------------------------------------------------
    text(r"""
        ## Execution State 1: Leakage-Free OOD Splitting & Train-Only Omics Standardization
        - **What is happening**: Partition data into `Random`, `Leave-Cell-Out` (`src/splitting.py`), and `Scaffold` splits, then fit `StandardScaler`, `PCA`, and `MGMT` imputation (`src/standardization.py`) strictly on training cell lines.
        - **What changes**: Held-out test cell lines (`GAMG`, `LN-229`) and held-out scaffolds (`Temozolomide`, `Irinotecan`) have zero overlap with training statistics.
        - **Why it matters**: Prevents test-set gene covariance and drug potency priors from leaking into training.
    """)
    link(trace_state_1_leakage_free_preprocessing)
    note("""
        [2:30 - 4:30] Step into `trace_state_1_leakage_free_preprocessing()`.
        Verify in the Variable Panel that `leakage_audit_passed` is True.
    """)

    state_1_state = trace_state_1_leakage_free_preprocessing(SAMPLE_CELL_LINES, SAMPLE_DRUGS)  # @inspect state_1_state

    # -------------------------------------------------------------------------
    # Execution State 2 (Minutes 4:30 - 6:15): Two-Tower MLP Forward Pass
    # -------------------------------------------------------------------------
    text(r"""
        ## Execution State 2: Two-Tower `BaselineMLP` Forward Pass & Latent Fusion
        - **What is happening**: Trace a single forward pass for **Temozolomide (TMZ)** on two GBM cell lines with opposite `MGMT` status: **U-87 MG** (`MGMT_Expr = 0.42`, sensitive) vs. **T98G** (`MGMT_Expr = 5.84`, resistant).
        - **What changes**: Drug and cell towers project inputs into $\mathbf{h}_{\text{drug}}$ and $\mathbf{h}_{\text{cell}}$, concatenate into $\mathbf{z}_{\text{joint}}$, and predict de-standardized $\widehat{\text{LN\_IC50}}$.
        - **Why it matters**: Shows how the cell tower captures `MGMT`-driven relative resistance ($+1.258$ log units higher $\text{IC}_{50}$ in `T98G`) within a fixed drug.
    """)
    link(trace_state_2_two_tower_forward_pass)
    note("""
        [4:30 - 6:15] Step into `trace_state_2_two_tower_forward_pass()`.
        Compare `u87_mgmt_low_pred_ic50` vs `t98g_mgmt_high_pred_ic50` in the Variable Panel.
    """)

    tmz_drug = SAMPLE_DRUGS[0]          # @inspect tmz_drug
    u87_cell = SAMPLE_CELL_LINES[0]     # @inspect u87_cell
    t98g_cell = SAMPLE_CELL_LINES[1]    # @inspect t98g_cell

    state_2_forward = trace_state_2_two_tower_forward_pass(tmz_drug, u87_cell, t98g_cell, DEMO_MLP_WEIGHTS)  # @inspect state_2_forward
    tmz_drug                            # @clear tmz_drug
    u87_cell                            # @clear u87_cell
    t98g_cell                           # @clear t98g_cell

    # -------------------------------------------------------------------------
    # Execution State 3 (Minutes 6:15 - 7:45): Shortcut Learning Diagnosis
    # -------------------------------------------------------------------------
    text(r"""
        ## Execution State 3: Shortcut Learning Diagnosis via Additive Baseline Decomposition
        - **What is happening**: Compare `BaselineMLP` ($0.8744$) against closed-form statistical baselines (`src/baselines.py`): `Cell-Only` ($0.2107$), `Drug-Only` ($0.8349$), and `Additive` ($\bar{y}_d + \bar{y}_c - \bar{y}_{\text{global}} = 0.8752$).
        - **What changes**: Nonlinear interaction gain $\Delta r_s = r_s(\text{MLP}) - r_s(\text{Additive})$ is **$-0.0008$** globally and **$-0.0013$** within-cell; $95.5\%$ of explainable rank variance comes from per-drug mean potency.
        - **Why it matters**: Proves that high random-split Spearman is an illusion driven by drug-mean memorization rather than drug–gene interaction learning.
    """)
    link(trace_state_3_shortcut_decomposition)
    note("""
        [6:15 - 7:45] Step into `trace_state_3_shortcut_decomposition()`.
        Highlight `interaction_gain_global` (-0.0008) in the Variable Panel.
    """)

    state_3_shortcut = trace_state_3_shortcut_decomposition(BENCHMARK_RESULTS)  # @inspect state_3_shortcut

    table(
        [
            {
                "Split Regime": "Random (In-Dist)",
                "Cell-Only": 0.2107,
                "Drug-Only": 0.8349,
                "Additive (No Inter.)": 0.8752,
                "BaselineMLP Global": "0.8744 ± 0.0051",
                "Within-Cell MLP": "0.8541 ± 0.0576",
                "Within-Cell Drug-Only": "0.8554 ± 0.0598",
            },
            {
                "Split Regime": "Leave-Cell-Out (Bio OOD)",
                "Cell-Only": None,
                "Drug-Only": 0.8362,
                "Additive (No Inter.)": None,
                "BaselineMLP Global": "0.6947 ± 0.2979*",
                "Within-Cell MLP": "0.8460 ± 0.0608",
                "Within-Cell Drug-Only": "0.8449 ± 0.0602",
            },
            {
                "Split Regime": "Scaffold (Chem OOD)",
                "Cell-Only": 0.2494,
                "Drug-Only": None,
                "Additive (No Inter.)": None,
                "BaselineMLP Global": "0.2905 ± 0.0045",
                "Within-Cell MLP": "0.1735 ± 0.0813",
                "Within-Cell Drug-Only": None,
            },
        ],
        caption="Table 1: Multi-Split Spearman Benchmark Across 5 Random Seeds (5,107 GBM Pairs)",
    )

    # -------------------------------------------------------------------------
    # Execution State 4 (Minutes 7:45 - 9:00): Chemical OOD Collapse & Decoupling
    # -------------------------------------------------------------------------
    text(r"""
        ## Execution State 4: Chemical OOD Collapse, Variance Shrinkage & Decoupling
        - **What is happening**: Evaluate `BaselineMLP` on unseen Bemis–Murcko scaffolds (`Scaffold` split, $47$ held-out drugs) and audit prediction shrinkage (`src/uncertainty.py`).
        - **What changes**:
          1. **Variance Shrinkage**: $\sigma(\hat{y})/\sigma(y)$ drops from **$0.862 \to 0.663$**, pulling predictions toward the training mean ($\text{TMZ}$ bias $= -3.13$, $\text{Irinotecan}$ bias $= -3.78$) and collapsing global Spearman ($0.8744 \to 0.2905$).
          2. **Within-Drug Ranking Preservation**: Despite global collapse, **median within-drug Spearman across cell lines remains $0.530$** (vs. $0.500$ on Random and Leave-Cell-Out).
        - **Why it matters**: The cell tower still ranks patient sensitivity ordering for novel drugs, but the drug tower cannot calibrate absolute dose ($\text{IC}_{50}$) for unseen scaffolds.
    """)
    link(trace_state_4_ood_shrinkage_and_decoupling)
    note("""
        [7:45 - 9:00] Step into `trace_state_4_ood_shrinkage_and_decoupling()`.
        Contrast `global_spearman` (0.2905) with `within_drug_median_spearman` (0.530) on Scaffold OOD.
    """)

    state_4_ood = trace_state_4_ood_shrinkage_and_decoupling(BENCHMARK_RESULTS, SAMPLE_DRUGS)  # @inspect state_4_ood

    plot(build_ood_comparison_chart_spec())
    table(
        SCAFFOLD_NOVELTY_BINS,
        caption="Table 2: Chemical Scaffold Novelty (Max Tanimoto Similarity to Train) vs. Per-Drug RMSE",
    )

    # -------------------------------------------------------------------------
    # Conclusion (Minutes 9:00 - 10:00): Key Result & Healthcare Interpretation
    # -------------------------------------------------------------------------
    text(r"""
        ## 3. Conclusion & Healthcare Interpretation (Minutes 9:00 – 10:00)
        1. **Mandatory OOD & Baseline Auditing**: Preclinical drug-response AI must report `Additive` baselines alongside `Leave-Cell-Out` and `Scaffold` splits to rule out shortcut learning.
        2. **Relative Patient Stratification vs. Absolute Dosing**: Under chemical OOD, preserved within-drug Spearman ($0.530$) supports **relative patient stratification** (identifying which GBM tumors are most responsive to a candidate compound), whereas variance shrinkage ($\sigma(\hat{y})/\sigma(y)=0.663$) warns against using uncalibrated $\text{IC}_{50}$ predictions for clinical dosing.
    """)
    note("""
        [9:00 - 10:00] Conclude with the clinical takeaway: use the model for relative patient
        stratification on novel compounds, not uncalibrated absolute dosing.
    """)


def trace_state_1_leakage_free_preprocessing(cell_lines, drugs):
    """State 1: Partition cell lines/scaffolds and fit train-only omics standardization."""
    train_cells = [c.cell_line_name for c in cell_lines[:2]]
    heldout_test_cells = [c.cell_line_name for c in cell_lines[2:]]

    train_scaffolds = [d.murcko_scaffold for d in drugs if d.split_assignment == "train_scaffold"]
    heldout_test_scaffolds = [d.murcko_scaffold for d in drugs if d.split_assignment == "test_scaffold"]

    cell_overlap = len(set(train_cells) & set(heldout_test_cells))
    scaffold_overlap = len(set(train_scaffolds) & set(heldout_test_scaffolds))

    train_gene0_vals = [c.raw_expr_vector[0] for c in cell_lines[:2]]
    gene0_train_mean = round(sum(train_gene0_vals) / len(train_gene0_vals), 4)
    gene0_train_std = round(
        math.sqrt(sum((v - gene0_train_mean) ** 2 for v in train_gene0_vals) / len(train_gene0_vals)),
        4,
    )

    gamg_raw_gene0 = cell_lines[2].raw_expr_vector[0]
    gamg_z_gene0 = round((gamg_raw_gene0 - gene0_train_mean) / gene0_train_std, 4)

    leakage_audit_passed = (cell_overlap == 0) and (scaffold_overlap == 0)
    return {
        "train_cells": train_cells,
        "heldout_test_cells": heldout_test_cells,
        "heldout_test_drugs": [d.drug_name for d in drugs if d.split_assignment == "test_scaffold"],
        "gene0_train_mean": gene0_train_mean,
        "gamg_z_gene0_standardized": gamg_z_gene0,
        "leakage_audit_passed": leakage_audit_passed,
    }


def _linear_relu(x_vec, weight_matrix, bias_vec):
    out = []
    for row_w, b in zip(weight_matrix, bias_vec):
        z = sum(w * x for w, x in zip(row_w, x_vec)) + b
        out.append(round(max(0.0, z), 4))
    return out


def trace_state_2_two_tower_forward_pass(drug, sensitive_cell, resistant_cell, weights):
    """State 2: Step through Drug Tower, Cell Tower, Concatenation, and IC50 De-standardization."""
    drug_fp = drug.morgan_fp_8d
    drug_emb = _linear_relu(drug_fp, weights["W_drug_1"], weights["b_drug_1"])  # @stepover

    u87_feat = sensitive_cell.pca_omics_vector + [sensitive_cell.mutations["PTEN"], sensitive_cell.mgmt_expr]
    t98g_feat = resistant_cell.pca_omics_vector + [resistant_cell.mutations["PTEN"], resistant_cell.mgmt_expr]

    u87_cell_emb = _linear_relu(u87_feat, weights["W_cell_1"], weights["b_cell_1"])    # @stepover
    t98g_cell_emb = _linear_relu(t98g_feat, weights["W_cell_1"], weights["b_cell_1"])  # @stepover

    joint_u87 = drug_emb + u87_cell_emb
    joint_t98g = drug_emb + t98g_cell_emb

    w_head = weights["w_head_random"]
    b_head = weights["b_head_random"]
    z_u87 = round(sum(w * h for w, h in zip(w_head, joint_u87)) + b_head, 4)
    z_t98g = round(sum(w * h for w, h in zip(w_head, joint_t98g)) + b_head, 4)

    u87_pred_ic50 = round(z_u87 * weights["y_std"] + weights["y_mean"], 3)
    t98g_pred_ic50 = round(z_t98g * weights["y_std"] + weights["y_mean"], 3)
    mgmt_resistance_shift = round(t98g_pred_ic50 - u87_pred_ic50, 3)

    return {
        "drug": drug.drug_name,
        "drug_emb_4d": drug_emb,
        "u87_mgmt_low_pred_ic50": u87_pred_ic50,
        "t98g_mgmt_high_pred_ic50": t98g_pred_ic50,
        "mgmt_resistance_shift_ln_ic50": mgmt_resistance_shift,
    }


def trace_state_3_shortcut_decomposition(benchmarks):
    """State 3: Quantify shortcut learning by comparing BaselineMLP to Additive drug+cell means."""
    rand = benchmarks["random"]
    sp_cell_only = rand["cell_only_spearman"]
    sp_drug_only = rand["drug_only_spearman"]
    sp_additive = rand["additive_spearman"]
    sp_mlp_global = rand["mlp_global_spearman_mean"]

    interaction_gain_global = round(sp_mlp_global - sp_additive, 4)
    interaction_gain_within_cell = round(
        rand["mlp_within_cell_spearman_mean"] - rand["drug_only_within_cell_spearman_mean"], 4
    )
    drug_potency_variance_share = round((sp_drug_only / sp_mlp_global) * 100.0, 1)

    return {
        "cell_only_spearman": sp_cell_only,
        "drug_only_spearman": sp_drug_only,
        "additive_baseline_spearman": sp_additive,
        "mlp_global_spearman": sp_mlp_global,
        "interaction_gain_global": interaction_gain_global,
        "interaction_gain_within_cell": interaction_gain_within_cell,
        "drug_potency_share_pct": f"{drug_potency_variance_share}%",
    }


def trace_state_4_ood_shrinkage_and_decoupling(benchmarks, drugs):
    """State 4: Contrast Global Spearman collapse with preserved Within-Drug Spearman under Scaffold OOD."""
    rand = benchmarks["random"]
    cell_ood = benchmarks["leave_cell"]
    scaf_ood = benchmarks["scaffold"]

    global_spearman_by_split = {
        "random": rand["mlp_global_spearman_mean"],
        "leave_cell_all_seeds": cell_ood["mlp_global_spearman_mean"],
        "leave_cell_stable_4seeds": cell_ood["mlp_stable_seeds_spearman_mean"],
        "scaffold_chemical_ood": scaf_ood["mlp_global_spearman_mean"],
    }

    shrinkage_ratios = {
        "random": rand["shrinkage_ratio"],
        "leave_cell": cell_ood["shrinkage_ratio"],
        "scaffold": scaf_ood["shrinkage_ratio"],
    }

    within_drug_median_spearman = {
        "random": rand["per_drug_median_spearman"],
        "leave_cell": cell_ood["per_drug_median_spearman"],
        "scaffold": scaf_ood["per_drug_median_spearman"],
    }

    tmz = drugs[0]
    irinotecan = drugs[1]
    tmz_scaffold_bias = round(tmz.pred_mean_ln_ic50_scaffold - tmz.true_mean_ln_ic50, 2)
    irinotecan_scaffold_bias = round(
        irinotecan.pred_mean_ln_ic50_scaffold - irinotecan.true_mean_ln_ic50, 2
    )

    return {
        "global_spearman": global_spearman_by_split,
        "shrinkage_ratio_pred_over_true_std": shrinkage_ratios,
        "within_drug_median_spearman": within_drug_median_spearman,
        "tmz_scaffold_bias_ln_ic50": tmz_scaffold_bias,
        "irinotecan_scaffold_bias_ln_ic50": irinotecan_scaffold_bias,
        "scaffold_large_error_fraction": f"{scaf_ood['large_error_rate_gt_2log'] * 100:.1f}%",
    }


def build_ood_comparison_chart_spec():
    """Interactive Vega-Lite grouped bar chart comparing Global vs. Within-Drug Spearman."""
    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": "Global Collapse vs. Within-Drug Ranking Preservation Across OOD Splits",
        "width": 460,
        "height": 260,
        "data": {
            "values": [
                {"Split": "1. Random (In-Dist)", "Metric": "Drug-Only Baseline", "Spearman": 0.8349},
                {"Split": "1. Random (In-Dist)", "Metric": "MLP Global Spearman", "Spearman": 0.8744},
                {"Split": "1. Random (In-Dist)", "Metric": "MLP Within-Drug Median", "Spearman": 0.5000},
                {"Split": "2. Leave-Cell (Bio OOD)", "Metric": "Drug-Only Baseline", "Spearman": 0.8362},
                {"Split": "2. Leave-Cell (Bio OOD)", "Metric": "MLP Global Spearman", "Spearman": 0.8436},
                {"Split": "2. Leave-Cell (Bio OOD)", "Metric": "MLP Within-Drug Median", "Spearman": 0.5000},
                {"Split": "3. Scaffold (Chem OOD)", "Metric": "Drug-Only Baseline", "Spearman": 0.0000},
                {"Split": "3. Scaffold (Chem OOD)", "Metric": "MLP Global Spearman", "Spearman": 0.2905},
                {"Split": "3. Scaffold (Chem OOD)", "Metric": "MLP Within-Drug Median", "Spearman": 0.5300},
            ]
        },
        "mark": {"type": "bar", "cornerRadiusTopLeft": 3, "cornerRadiusTopRight": 3},
        "encoding": {
            "x": {"field": "Split", "type": "nominal", "title": "Evaluation Split Regime", "axis": {"labelAngle": 0}},
            "y": {"field": "Spearman", "type": "quantitative", "title": "Spearman Rank Correlation", "scale": {"domain": [0, 1.0]}},
            "xOffset": {"field": "Metric"},
            "color": {
                "field": "Metric",
                "type": "nominal",
                "scale": {"range": ["#9e9e9e", "#1976d2", "#d32f2f"]},
            },
        },
    }


if __name__ == "__main__":
    main()
