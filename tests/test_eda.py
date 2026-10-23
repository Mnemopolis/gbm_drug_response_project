"""Unit tests exclusively for `src.eda`."""

import tempfile
import unittest
from pathlib import Path

from src.eda import compute_cohort_eda_summary, plot_label_distribution_across_splits
from src.splitting import partition_dataframe_by_split
from tests.fixtures.synthetic_data import make_synthetic_merged_dataframe


class TestEDA(unittest.TestCase):
    """Verify cohort summary statistics and 3-split LN_IC50 histogram plotting."""

    def setUp(self) -> None:
        self.df = make_synthetic_merged_dataframe(n_cells=6, n_drugs=8, seed=42)

    def test_compute_cohort_eda_summary(self) -> None:
        summary = compute_cohort_eda_summary(self.df)
        self.assertEqual(summary["n_unique_cells"], 6)
        self.assertEqual(summary["n_unique_drugs"], 10)
        self.assertEqual(summary["n_records"], 60)
        self.assertAlmostEqual(summary["mgmt_missing_rate"], 0.0, places=5)

    def test_plot_label_distribution_across_splits(self) -> None:
        splits_dict = {
            s: partition_dataframe_by_split(self.df, s, test_size=0.25)
            for s in ["random", "leave_cell", "scaffold"]
        }
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "three_split.png"
            saved_path = plot_label_distribution_across_splits(splits_dict, out_path)
            self.assertTrue(saved_path.exists())
            self.assertGreater(saved_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
