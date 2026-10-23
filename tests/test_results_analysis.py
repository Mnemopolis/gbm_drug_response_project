"""Unit tests exclusively for `src.results_analysis`."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.results_analysis import (
    compute_per_cell_error_stats,
    compute_per_drug_error_stats,
    find_gbm_drugs_in_top_errors,
    get_top_k_prediction_errors,
    inspect_key_gbm_drugs,
    plot_per_drug_rmse_boxplot_across_splits,
    plot_prediction_bias_boxplot_across_splits,
    plot_scatter_predictions_three_splits,
)
from src.splitting import partition_dataframe_by_split
from src.standardization import fit_transform_cell_pca_features
from tests.fixtures.synthetic_data import make_synthetic_merged_dataframe


class TestResultsAnalysis(unittest.TestCase):
    """Verify Top-K error tables, per-drug/per-cell stats, GBM drug lookup, and plots."""

    def setUp(self) -> None:
        df = make_synthetic_merged_dataframe(n_cells=8, n_drugs=10, seed=42)
        expr_cols = [c for c in df.columns if c.startswith("EXPR_")]
        mut_cols = [c for c in df.columns if c.startswith("MUT_")]
        self.preds_store = {}
        rng = np.random.default_rng(42)
        for split in ["random", "leave_cell", "scaffold"]:
            tr_raw, te_raw = partition_dataframe_by_split(df, split, test_size=0.25)
            _, te_pca = fit_transform_cell_pca_features(
                tr_raw, te_raw, expr_cols, mut_cols, ["MGMT_Expr"], n_pca=4
            )
            te_pca = te_pca.reset_index(drop=True)
            trues = te_pca["LN_IC50"].values
            preds = trues + rng.normal(0, 0.35, len(trues))
            self.preds_store[split] = {
                "test_df": te_pca,
                "preds": preds,
                "trues": trues,
                "spearman": 0.78,
                "rmse": 0.36,
            }

    def test_error_tables_and_gbm_drug_inspection(self) -> None:
        top5 = get_top_k_prediction_errors(self.preds_store["random"], k=5)
        pstats = compute_per_drug_error_stats(self.preds_store["random"], min_n=1)
        cstats = compute_per_cell_error_stats(self.preds_store["random"], min_n=1)
        hits = find_gbm_drugs_in_top_errors(pstats, top_n=20)
        gbm_df = inspect_key_gbm_drugs(self.preds_store)

        self.assertEqual(len(top5), 5)
        self.assertGreater(len(pstats), 0)
        self.assertGreater(len(cstats), 0)
        self.assertIsInstance(hits, list)
        self.assertGreater(len(gbm_df), 0)

    def test_diagnostic_plot_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            p1 = plot_scatter_predictions_three_splits(
                self.preds_store, tmp_dir / "scatter.png"
            )
            p2 = plot_per_drug_rmse_boxplot_across_splits(
                self.preds_store, tmp_dir / "rmse.png"
            )
            p3 = plot_prediction_bias_boxplot_across_splits(
                self.preds_store, tmp_dir / "bias.png"
            )
            self.assertTrue(p1.exists() and p2.exists() and p3.exists())


if __name__ == "__main__":
    unittest.main()
