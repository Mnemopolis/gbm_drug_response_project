"""Unit tests exclusively for `src.splitting`."""

import unittest

from src.splitting import partition_dataframe_by_split, partition_train_validation
from tests.fixtures.synthetic_data import make_synthetic_merged_dataframe


class TestSplitting(unittest.TestCase):
    """Verify Random, Leave-Cell-Out, and Scaffold splitting strategies."""

    def setUp(self) -> None:
        self.df = make_synthetic_merged_dataframe(n_cells=10, n_drugs=12, seed=42)

    def test_random_split_proportions(self) -> None:
        tr, te = partition_dataframe_by_split(self.df, "random", test_size=0.20)
        self.assertEqual(len(tr) + len(te), len(self.df))
        self.assertAlmostEqual(len(te) / len(self.df), 0.20, delta=0.05)

    def test_leave_cell_split_disjoint_cells(self) -> None:
        tr, te = partition_dataframe_by_split(self.df, "leave_cell", test_size=0.20)
        tr_cells = set(tr["ModelID"].unique())
        te_cells = set(te["ModelID"].unique())
        self.assertEqual(len(tr_cells.intersection(te_cells)), 0)

    def test_scaffold_split_disjoint_scaffolds(self) -> None:
        tr, te = partition_dataframe_by_split(self.df, "scaffold", test_size=0.25)
        tr_scaf = set(tr["Scaffold"].unique())
        te_scaf = set(te["Scaffold"].unique())
        self.assertEqual(len(tr_scaf.intersection(te_scaf)), 0)

    def test_partition_train_validation_stratification(self) -> None:
        tr, val = partition_train_validation(self.df, val_size=0.20, seed=42)
        self.assertEqual(tr["ModelID"].nunique(), self.df["ModelID"].nunique())
        self.assertEqual(len(tr) + len(val), len(self.df))


if __name__ == "__main__":
    unittest.main()
