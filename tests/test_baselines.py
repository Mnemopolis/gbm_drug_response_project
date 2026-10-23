"""Unit tests exclusively for `src.baselines`."""

import unittest

import numpy as np

from src.baselines import (
    compute_within_cell_spearman,
    evaluate_all_statistical_baselines,
    evaluate_nearest_drug_scaffold_baseline,
)
from src.splitting import partition_dataframe_by_split
from tests.fixtures.synthetic_data import make_synthetic_merged_dataframe


class TestBaselines(unittest.TestCase):
    """Verify Cell-only, Drug-only, Additive, Within-Cell, and Nearest-Drug baselines."""

    def setUp(self) -> None:
        self.df = make_synthetic_merged_dataframe(n_cells=8, n_drugs=10, seed=42)

    def test_evaluate_all_statistical_baselines_across_regimes(self) -> None:
        tr_rand, te_rand = partition_dataframe_by_split(
            self.df, "random", test_size=0.20
        )
        res_rand = evaluate_all_statistical_baselines(tr_rand, te_rand)
        self.assertGreater(res_rand["drug_only"], 0.5)
        self.assertGreater(res_rand["additive"], 0.5)

        tr_scaf, te_scaf = partition_dataframe_by_split(
            self.df, "scaffold", test_size=0.25
        )
        res_scaf = evaluate_all_statistical_baselines(tr_scaf, te_scaf)
        self.assertTrue(np.isnan(res_scaf["drug_only"]))

    def test_evaluate_nearest_drug_scaffold_baseline(self) -> None:
        tr_scaf, te_scaf = partition_dataframe_by_split(
            self.df, "scaffold", test_size=0.25
        )
        sp_nn = evaluate_nearest_drug_scaffold_baseline(tr_scaf, te_scaf)
        self.assertIsInstance(sp_nn, float)
        self.assertFalse(np.isnan(sp_nn))

    def test_compute_within_cell_spearman(self) -> None:
        trues = np.array([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])
        preds = np.array([1.1, 1.9, 3.2, 4.1, 12.0, 18.0, 29.0, 41.0])
        cells = np.array(["C1", "C1", "C1", "C1", "C2", "C2", "C2", "C2"])
        mean_sp, _, n_cells = compute_within_cell_spearman(preds, trues, cells, min_n=3)
        self.assertEqual(n_cells, 2)
        self.assertGreater(mean_sp, 0.95)


if __name__ == "__main__":
    unittest.main()
