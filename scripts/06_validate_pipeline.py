"""
End-to-End Pipeline Validation & Anti-Leakage Audit Smoke Test.

Verifies all core domain modules (`deduplication`, `splitting`, `standardization`,
`pipeline_audit`, `eda`, `baselines`, `uncertainty`, `results_analysis`) on
synthetic data in <2 seconds without requiring multi-GB raw CSV files.
"""

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.deduplication import deduplicate_gdsc_drug_cell_pairs
from src.eda import compute_cohort_eda_summary, plot_label_distribution_across_splits
from src.pipeline_audit import run_full_pipeline_audit
from src.results_analysis import (
    compute_per_cell_error_stats,
    compute_per_drug_error_stats,
    get_top_k_prediction_errors,
    plot_per_drug_rmse_boxplot_across_splits,
    plot_prediction_bias_boxplot_across_splits,
    plot_scatter_predictions_three_splits,
)
from src.standardization import standardize_target_labels
from src.uncertainty import (
    compare_cell_performance_between_seeds,
    summarize_prediction_shrinkage_and_bias,
    summarize_seed_stability_across_splits,
)
from tests.fixtures.synthetic_data import make_synthetic_pca_splits


def main() -> None:
    """Execute end-to-end smoke validation across all core modules."""
    print("=" * 70)
    print("Running End-to-End Pipeline Validation & Anti-Leakage Audit")
    print("=" * 70)

    bundle = make_synthetic_pca_splits(n_cells=10, n_drugs=10, n_pca=5, seed=42)
    synth_df = bundle["raw_df"]
    splits_dict = bundle["splits_dict"]
    preds_store = bundle["preds_store"]
    results_mock = bundle["results_mock"]

    dedup_df = deduplicate_gdsc_drug_cell_pairs(synth_df)
    eda_info = compute_cohort_eda_summary(synth_df)
    assert len(dedup_df) == len(synth_df) and eda_info["n_unique_cells"] == 10
    print("  [1/5] Deduplication & EDA summary verified.")

    audit_df = run_full_pipeline_audit(splits_dict)
    assert audit_df["passed"].all()
    print("  [2/5] Splitting, train-only PCA standardization & anti-leakage audit verified.")

    y_scaled, _, _ = standardize_target_labels(synth_df["LN_IC50"].values)
    assert abs(float(y_scaled.mean())) < 1e-5 and abs(float(y_scaled.std()) - 1.0) < 1e-5
    print("  [3/5] Target Z-score standardization verified.")

    stab_df = summarize_seed_stability_across_splits(results_mock)
    shrink_df = summarize_prediction_shrinkage_and_bias(preds_store)
    collapse_df = compare_cell_performance_between_seeds(
        preds_store["leave_cell"]["preds"],
        preds_store["leave_cell"]["preds"] * 0.1,
        preds_store["leave_cell"]["trues"],
        preds_store["leave_cell"]["test_df"]["ModelID"].values,
    )
    assert len(stab_df) == 3 and len(shrink_df) == 3 and len(collapse_df) > 0
    print("  [4/5] Multi-seed stability, Seed 2 collapse diagnosis & shrinkage verified.")

    top_err = get_top_k_prediction_errors(preds_store["random"], k=5)
    pstats = compute_per_drug_error_stats(preds_store["random"], min_n=1)
    cstats = compute_per_cell_error_stats(preds_store["random"], min_n=2)
    assert len(top_err) == 5 and len(pstats) > 0 and len(cstats) > 0

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        plot_label_distribution_across_splits(splits_dict, tmp / "three_split.png")
        plot_scatter_predictions_three_splits(preds_store, tmp / "scatter.png")
        plot_per_drug_rmse_boxplot_across_splits(preds_store, tmp / "rmse.png")
        plot_prediction_bias_boxplot_across_splits(preds_store, tmp / "bias.png")
        assert (tmp / "three_split.png").exists() and (tmp / "scatter.png").exists()
    print("  [5/5] Error analysis tables and Matplotlib figure generation verified.")
    print("=" * 70)
    print("SUCCESS: All 17 core src/ modules and pipeline stages passed validation!")
    print("=" * 70)


if __name__ == "__main__":
    main()
