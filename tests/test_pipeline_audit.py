"""Unit tests exclusively for `src.pipeline_audit`."""

import unittest

from src.pipeline_audit import (
    audit_feature_matrix_integrity,
    audit_split_no_leakage,
    run_full_pipeline_audit,
)
from src.splitting import partition_dataframe_by_split
from src.standardization import fit_transform_cell_pca_features
from tests.fixtures.synthetic_data import make_synthetic_merged_dataframe


class TestPipelineAudit(unittest.TestCase):
    """Verify programmatic anti-leakage assertions and feature matrix integrity checks."""

    def setUp(self) -> None:
        self.df = make_synthetic_merged_dataframe(n_cells=10, n_drugs=12, seed=42)
        self.expr_cols = [c for c in self.df.columns if c.startswith("EXPR_")]
        self.mut_cols = [c for c in self.df.columns if c.startswith("MUT_")]
        self.mgmt_cols = ["MGMT_Expr"]

    def test_run_full_pipeline_audit_passes_clean_splits(self) -> None:
        splits_dict = {}
        for split in ["random", "leave_cell", "scaffold"]:
            tr_raw, te_raw = partition_dataframe_by_split(
                self.df, split, test_size=0.25
            )
            splits_dict[split] = fit_transform_cell_pca_features(
                tr_raw, te_raw, self.expr_cols, self.mut_cols, self.mgmt_cols, n_pca=4
            )
        audit_df = run_full_pipeline_audit(splits_dict)
        self.assertEqual(len(audit_df), 3)
        self.assertTrue(audit_df["passed"].all())

    def test_audit_split_no_leakage_raises_on_cell_overlap(self) -> None:
        tr_raw, te_raw = partition_dataframe_by_split(
            self.df, "leave_cell", test_size=0.25
        )
        bad_test = te_raw.copy()
        bad_test.iloc[0, bad_test.columns.get_loc("ModelID")] = tr_raw["ModelID"].iloc[
            0
        ]
        with self.assertRaises(AssertionError):
            audit_split_no_leakage(tr_raw, bad_test, "leave_cell")

    def test_audit_feature_matrix_integrity_rejects_unreduced_expr(self) -> None:
        with self.assertRaises(AssertionError):
            audit_feature_matrix_integrity(self.df, "random")


if __name__ == "__main__":
    unittest.main()
