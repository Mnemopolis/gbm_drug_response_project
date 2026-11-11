"""
Unit tests for `src/results_analysis.py`.
"""

import tempfile
import unittest
from pathlib import Path

from src.results_analysis import (
    compute_per_cell_error_stats,
    compute_per_drug_error_stats,
    find_gbm_drugs_in_top_errors,
    get_top_k_prediction_errors,
    inspect_key_gbm_drugs,
    plot_global_vs_within_drug_spearman,
    plot_per_drug_rmse_boxplot_across_splits,
    plot_per_drug_spearman_scaffold_boxplot,
    plot_prediction_bias_boxplot_across_splits,
    plot_random_split_baseline_decomposition,
    plot_scatter_predictions_three_splits,
    plot_three_split_ood_comparison,
)
from tests.fixtures.synthetic_data import make_synthetic_pca_splits


class TestResultsAnalysis(unittest.TestCase):
    """Test error breakdown tables and diagnostic plots using shared synthetic PCA splits."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = make_synthetic_pca_splits(n_cells=8, n_drugs=10, n_pca=4, seed=42)
        cls.preds_store = cls.bundle["preds_store"]
        cls.results_mock = cls.bundle["results_mock"]

    def test_error_tables_and_gbm_drug_inspection(self) -> None:
        top5 = get_top_k_prediction_errors(self.preds_store["random"], k=5)
        self.assertEqual(len(top5), 5)

        pstats = compute_per_drug_error_stats(self.preds_store["random"], min_n=1)
        self.assertGreater(len(pstats), 0)
        self.assertIsInstance(find_gbm_drugs_in_top_errors(pstats, top_n=5), list)

        cstats = compute_per_cell_error_stats(self.preds_store["random"], min_n=2)
        self.assertGreater(len(cstats), 0)

        gbm_df = inspect_key_gbm_drugs(self.preds_store, drug_keywords=["Temozolomide"])
        self.assertGreater(len(gbm_df), 0)

    def test_diagnostic_plot_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            paths = [
                plot_random_split_baseline_decomposition(self.results_mock["random"], tmp / "f1.png"),
                plot_three_split_ood_comparison(self.results_mock, tmp / "f2.png"),
                plot_scatter_predictions_three_splits(self.preds_store, tmp / "f3.png"),
                plot_per_drug_spearman_scaffold_boxplot(self.preds_store, tmp / "f4.png"),
                plot_per_drug_rmse_boxplot_across_splits(self.preds_store, tmp / "f5.png"),
                plot_prediction_bias_boxplot_across_splits(self.preds_store, tmp / "f6.png"),
                plot_global_vs_within_drug_spearman(self.preds_store, tmp / "f7.png"),
            ]
            for p in paths:
                self.assertTrue(p.is_file())
