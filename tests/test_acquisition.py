"""Unit tests exclusively for `src.acquisition`."""

import tempfile
import unittest
from pathlib import Path

from src.acquisition import (
    convert_and_extract_methylation_features,
    extract_expression_features,
    extract_mutation_features,
    load_depmap_model_mapping,
    load_raw_gdsc_gbm_records,
)
from tests.fixtures.synthetic_data import write_synthetic_raw_csv_files


class TestAcquisition(unittest.TestCase):
    """Verify raw GDSC2, DepMap expression, somatic mutation, and RRBS methylation ingestion."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.paths = write_synthetic_raw_csv_files(Path(self.tmp_dir.name))

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_load_depmap_model_mapping(self) -> None:
        mapping = load_depmap_model_mapping(self.paths["model"])
        self.assertEqual(len(mapping), 4)
        self.assertIn("ModelID", mapping.columns)
        self.assertIn("SangerModelID", mapping.columns)

    def test_load_raw_gdsc_gbm_records_filters_glioblastoma(self) -> None:
        gbm_df = load_raw_gdsc_gbm_records(self.paths["gdsc"], self.paths["model"])
        self.assertEqual(len(gbm_df), 4)
        self.assertTrue((gbm_df["CANCER_TYPE"] == "Glioblastoma").all())

    def test_extract_expression_features_and_mgmt_proxy(self) -> None:
        expr_gbm, mgmt_expr = extract_expression_features(
            self.paths["expr"], valid_cells=["ACH-000001", "ACH-000002"]
        )
        self.assertEqual(expr_gbm.shape, (2, 3))
        self.assertEqual(list(mgmt_expr.columns), ["MGMT_Expr"])

    def test_extract_mutation_features_drops_zero_columns_and_non_default(self) -> None:
        mut_gbm = extract_mutation_features(
            self.paths["mut"], valid_cells=["ACH-000001", "ACH-000002", "ACH-000003"]
        )
        self.assertEqual(mut_gbm.shape, (3, 2))
        self.assertNotIn("ZERO_GENE", mut_gbm.columns)

    def test_convert_and_extract_methylation_features(self) -> None:
        mgmt_methyl = convert_and_extract_methylation_features(
            self.paths["methyl_txt"], self.paths["methyl_csv"], self.paths["model"]
        )
        self.assertTrue(self.paths["methyl_csv"].exists())
        self.assertIsNotNone(mgmt_methyl)
        self.assertEqual(len(mgmt_methyl), 2)


if __name__ == "__main__":
    unittest.main()
