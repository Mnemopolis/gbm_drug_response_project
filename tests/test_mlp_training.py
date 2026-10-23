"""Unit tests exclusively for `src.mlp_training`."""

import unittest

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.splitting import partition_dataframe_by_split
from src.standardization import fit_transform_cell_pca_features
from tests.fixtures.synthetic_data import make_synthetic_merged_dataframe

if TORCH_AVAILABLE:
    from src.mlp_training import train_single_seed


class TestMlpTraining(unittest.TestCase):
    """Verify single-seed training loop, validation early stopping, and metric output."""

    @unittest.skipUnless(
        TORCH_AVAILABLE, "PyTorch is not installed in this environment"
    )
    def test_train_single_seed_execution(self) -> None:
        df = make_synthetic_merged_dataframe(n_cells=6, n_drugs=6, seed=42)
        expr_cols = [c for c in df.columns if c.startswith("EXPR_")]
        mut_cols = [c for c in df.columns if c.startswith("MUT_")]
        tr_raw, te_raw = partition_dataframe_by_split(df, "random", test_size=0.25)
        tr_pca, te_pca = fit_transform_cell_pca_features(
            tr_raw, te_raw, expr_cols, mut_cols, ["MGMT_Expr"], n_pca=3
        )
        metrics = train_single_seed(
            tr_pca, te_pca, te_pca, seed=0, epochs=2, patience=2, batch_size=8
        )
        self.assertIn("test_rmse", metrics)
        self.assertIn("test_spearman", metrics)
        self.assertEqual(len(metrics["test_preds"]), len(te_pca))


if __name__ == "__main__":
    unittest.main()
