"""Unit tests exclusively for `src.dataset` (tabular PyTorch Dataset and DataLoaders)."""

import unittest

from src.dataset import DrugResponseDataset, HAS_TORCH, create_dataloaders
from src.splitting import partition_dataframe_by_split, partition_train_validation
from src.standardization import fit_transform_cell_pca_features
from tests.fixtures.synthetic_data import make_synthetic_merged_dataframe


class TestDataset(unittest.TestCase):
    """Verify PyTorch Dataset and DataLoader construction."""

    def setUp(self) -> None:
        df, expr_c, mut_c, mgmt_c = make_synthetic_merged_dataframe(
            n_cells=10, n_drugs=12, seed=42, return_cols=True
        )
        tr_raw, te_raw = partition_dataframe_by_split(df, "random", test_size=0.2)
        self.trainval_df, self.test_df = fit_transform_cell_pca_features(
            tr_raw, te_raw, expr_c, mut_c, mgmt_c, n_pca=4
        )
        self.train_df, self.val_df = partition_train_validation(self.trainval_df, val_size=0.2)

    @unittest.skipUnless(HAS_TORCH, "PyTorch is not installed in this environment")
    def test_drug_response_dataset_and_dataloaders(self) -> None:
        train_loader, val_loader, test_loader, train_ds = create_dataloaders(
            self.train_df, self.val_df, self.test_df, batch_size=8
        )
        self.assertEqual(len(train_ds), len(self.train_df))
        self.assertGreater(train_ds.cell_dim, 0)
        fp_batch, cell_batch, y_batch = next(iter(train_loader))
        self.assertEqual(fp_batch.shape[1], 2048)
        self.assertEqual(cell_batch.shape[1], train_ds.cell_dim)
        self.assertEqual(len(y_batch.shape), 1)


if __name__ == "__main__":
    unittest.main()
