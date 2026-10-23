"""Unit tests exclusively for `src.standardization`."""

import unittest

import numpy as np

from src.splitting import partition_dataframe_by_split
from src.standardization import (
    fit_transform_cell_pca_features,
    standardize_target_labels,
    unstandardize_target_labels,
)
from tests.fixtures.synthetic_data import make_synthetic_merged_dataframe


class TestStandardization(unittest.TestCase):
    """Verify leakage-free train-only PCA standardization and target Z-score scaling."""

    def setUp(self) -> None:
        self.df = make_synthetic_merged_dataframe(n_cells=10, n_drugs=10, seed=42)
        self.expr_cols = [c for c in self.df.columns if c.startswith("EXPR_")]
        self.mut_cols = [c for c in self.df.columns if c.startswith("MUT_")]
        self.mgmt_cols = ["MGMT_Expr"]

    def test_fit_transform_cell_pca_features_replaces_expr_columns(self) -> None:
        tr_raw, te_raw = partition_dataframe_by_split(
            self.df, "leave_cell", test_size=0.20
        )
        tr_pca, te_pca = fit_transform_cell_pca_features(
            tr_raw, te_raw, self.expr_cols, self.mut_cols, self.mgmt_cols, n_pca=5
        )
        pca_cols = [c for c in tr_pca.columns if c.startswith("PCA_")]
        self.assertEqual(len(pca_cols), 5)
        self.assertEqual(len([c for c in tr_pca.columns if c.startswith("EXPR_")]), 0)
        self.assertFalse(tr_pca[pca_cols].isna().any().any())
        self.assertFalse(te_pca[pca_cols].isna().any().any())

    def test_standardize_and_unstandardize_target_labels(self) -> None:
        y = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float32)
        y_scaled, mean, std = standardize_target_labels(y)
        self.assertAlmostEqual(float(y_scaled.mean()), 0.0, places=5)
        self.assertAlmostEqual(float(y_scaled.std()), 1.0, places=5)
        y_inv = unstandardize_target_labels(y_scaled, mean, std)
        np.testing.assert_allclose(y, y_inv, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
