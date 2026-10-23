"""Unit tests exclusively for `src.graph_data`."""

import unittest

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.graph_data import get_cell_feature_columns
from src.splitting import partition_dataframe_by_split
from src.standardization import fit_transform_cell_pca_features
from tests.fixtures.synthetic_data import make_synthetic_merged_dataframe

if TORCH_AVAILABLE:
    from src.graph_data import DrugResponseDataset, create_dataloaders


class TestGraphData(unittest.TestCase):
    """Verify cell feature column selection, PyTorch Dataset, and DataLoaders."""

    def setUp(self) -> None:
        df = make_synthetic_merged_dataframe(n_cells=8, n_drugs=8, seed=42)
        expr_cols = [c for c in df.columns if c.startswith("EXPR_")]
        mut_cols = [c for c in df.columns if c.startswith("MUT_")]
        tr_raw, te_raw = partition_dataframe_by_split(df, "random", test_size=0.25)
        self.tr_pca, self.te_pca = fit_transform_cell_pca_features(
            tr_raw, te_raw, expr_cols, mut_cols, ["MGMT_Expr"], n_pca=4
        )

    def test_get_cell_feature_columns(self) -> None:
        cols = get_cell_feature_columns(self.tr_pca)
        self.assertEqual(len(cols), 4 + 4 + 1)

    @unittest.skipUnless(
        TORCH_AVAILABLE, "PyTorch is not installed in this environment"
    )
    def test_drug_response_dataset_and_dataloaders(self) -> None:
        ds = DrugResponseDataset(self.tr_pca)
        self.assertEqual(len(ds), len(self.tr_pca))
        self.assertAlmostEqual(float(ds.labels.mean().item()), 0.0, places=5)

        tr_loader, _val_loader, _te_loader, _ = create_dataloaders(
            self.tr_pca, self.te_pca, self.te_pca, batch_size=8
        )
        fp_b, cell_b, y_b = next(iter(tr_loader))
        self.assertEqual(fp_b.shape[1], 2048)
        self.assertEqual(cell_b.shape[1], 9)
        self.assertEqual(len(y_b), 8)


if __name__ == "__main__":
    unittest.main()
