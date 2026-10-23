"""Unit tests exclusively for `src.baseline_workflow`."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.baseline_workflow import run_baseline_workflow
from src.splitting import partition_dataframe_by_split
from src.standardization import fit_transform_cell_pca_features
from tests.fixtures.synthetic_data import make_synthetic_merged_dataframe


class TestBaselineWorkflow(unittest.TestCase):
    """Verify Stage 2 baseline workflow orchestration over cached split pickles."""

    def test_run_baseline_workflow_returns_all_splits(self) -> None:
        df = make_synthetic_merged_dataframe(n_cells=10, n_drugs=12, seed=42)
        expr_cols = [c for c in df.columns if c.startswith("EXPR_")]
        mut_cols = [c for c in df.columns if c.startswith("MUT_")]

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            mock_paths = {}
            for split in ["random", "leave_cell", "scaffold"]:
                tr_raw, te_raw = partition_dataframe_by_split(df, split, test_size=0.25)
                tr_pca, te_pca = fit_transform_cell_pca_features(
                    tr_raw, te_raw, expr_cols, mut_cols, ["MGMT_Expr"], n_pca=4
                )
                tr_file = tmp_path / f"train_{split}.pkl"
                te_file = tmp_path / f"test_{split}.pkl"
                tr_pca.to_pickle(tr_file)
                te_pca.to_pickle(te_file)
                mock_paths[split] = {"train": tr_file, "test": te_file}

            with patch("src.baseline_workflow.get_split_data_paths", side_effect=lambda s: mock_paths[s]):
                summary = run_baseline_workflow()

        self.assertEqual(set(summary.keys()), {"random", "leave_cell", "scaffold"})
        self.assertIn("nearest_drug_spearman", summary["scaffold"])


if __name__ == "__main__":
    unittest.main()
