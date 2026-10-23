"""Unit tests exclusively for `src.uncertainty`."""

import unittest
import numpy as np
import pandas as pd

from src.uncertainty import (
    compare_cell_performance_between_seeds,
    summarize_prediction_shrinkage_and_bias,
    summarize_seed_stability_across_splits,
)


class TestUncertainty(unittest.TestCase):
    """Verify multi-seed stability audit, Seed 2 collapse diagnosis, and prediction shrinkage."""

    def test_summarize_seed_stability_detects_collapsed_seed(self) -> None:
        results_mock = {
            "random": {
                "model_global": {"values": [0.86, 0.87, 0.88, 0.87, 0.87]},
                "model_rmse": {"mean": 1.03, "values": [1.07, 1.02, 1.02, 1.02, 1.01]},
            },
            "leave_cell": {
                "model_global": {"values": [0.84, 0.83, 0.098, 0.85, 0.84]},
                "model_rmse": {"mean": 1.39, "values": [1.08, 1.10, 2.62, 1.08, 1.09]},
            },
        }
        df = summarize_seed_stability_across_splits(results_mock)
        self.assertFalse(bool(df.loc[df["split"] == "random", "has_collapsed_seed"].iloc[0]))
        self.assertTrue(bool(df.loc[df["split"] == "leave_cell", "has_collapsed_seed"].iloc[0]))

    def test_compare_cell_performance_between_seeds(self) -> None:
        trues = np.array([1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0])
        preds_s0 = np.array([1.1, 2.1, 2.9, 4.0, 2.1, 3.1, 3.9, 5.1])
        preds_s2 = np.array([3.0, 3.1, 3.0, 3.2, 3.1, 3.0, 3.2, 3.1])
        cells = np.array(["ACH-1", "ACH-1", "ACH-1", "ACH-1", "ACH-2", "ACH-2", "ACH-2", "ACH-2"])
        comp = compare_cell_performance_between_seeds(preds_s0, preds_s2, trues, cells)
        self.assertEqual(len(comp), 2)
        self.assertTrue((comp["spearman_drop"] > 0.3).all())

    def test_summarize_prediction_shrinkage_and_bias(self) -> None:
        preds_store = {
            "random": {"preds": np.array([1.0, 2.0, 3.0]), "trues": np.array([1.0, 2.0, 3.0])},
            "scaffold": {"preds": np.array([1.9, 2.0, 2.1]), "trues": np.array([0.0, 2.0, 4.0])},
        }
        shrink_df = summarize_prediction_shrinkage_and_bias(preds_store)
        scaf_ratio = float(shrink_df.loc[shrink_df["split"] == "scaffold", "std_ratio"].iloc[0])
        self.assertLess(scaf_ratio, 0.2)


if __name__ == "__main__":
    unittest.main()
