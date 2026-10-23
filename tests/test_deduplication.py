"""Unit tests exclusively for `src.deduplication`."""

import unittest

import pandas as pd

from src.deduplication import (
    deduplicate_drug_smiles_table,
    deduplicate_gdsc_drug_cell_pairs,
    merge_multimodal_dataset,
)
from tests.fixtures.synthetic_data import make_synthetic_merged_dataframe


class TestDeduplication(unittest.TestCase):
    """Verify replicate drug-cell IC50 averaging, SMILES deduplication, and multimodal merging."""

    def test_deduplicate_gdsc_drug_cell_pairs_averages_replicates(self) -> None:
        raw_df = pd.DataFrame(
            {
                "ModelID": ["ACH-001", "ACH-001", "ACH-002"],
                "DRUG_NAME": ["TMZ", "TMZ", "TMZ"],
                "LN_IC50": [2.0, 4.0, 5.0],
            }
        )
        dedup = deduplicate_gdsc_drug_cell_pairs(raw_df)
        self.assertEqual(len(dedup), 2)
        val = float(dedup.loc[dedup["ModelID"] == "ACH-001", "LN_IC50"].iloc[0])
        self.assertAlmostEqual(val, 3.0, places=5)

    def test_deduplicate_drug_smiles_table_normalizes_keys(self) -> None:
        raw_smiles = pd.DataFrame(
            {
                "drug_name": ["Drug-A", "Drug A", "Drug-B"],
                "canonical_smiles": ["CCO", "CCO", "c1ccccc1"],
            }
        )
        dedup_smiles = deduplicate_drug_smiles_table(raw_smiles)
        self.assertEqual(len(dedup_smiles), 2)
        self.assertIn("DRUG_KEY", dedup_smiles.columns)

    def test_merge_multimodal_dataset_prefixes_columns(self) -> None:
        synth = make_synthetic_merged_dataframe(n_cells=4, n_drugs=4)
        gdsc_grouped = synth[["ModelID", "DRUG_NAME", "LN_IC50"]].copy()
        drug_info = synth[
            ["DRUG_NAME", "SMILES", "DRUG_KEY", "Morgan_FP", "Scaffold"]
        ].drop_duplicates("DRUG_KEY")
        expr_gbm = pd.DataFrame(
            {"G1": [1.0, 2.0, 3.0, 4.0], "G2": [0.5, 1.5, 2.5, 3.5]},
            index=synth["ModelID"].unique(),
        )
        mut_gbm = pd.DataFrame(
            {"TP53": [1.0, 0.0, 1.0, 0.0]},
            index=synth["ModelID"].unique(),
        )
        mgmt_expr = pd.DataFrame(
            {"MGMT_Expr": [2.1, 3.1, 1.8, 2.9]},
            index=synth["ModelID"].unique(),
        )

        merged, expr_cols, mut_cols, mgmt_cols = merge_multimodal_dataset(
            gdsc_grouped, drug_info, expr_gbm, mut_gbm, mgmt_expr
        )
        self.assertEqual(len(merged), len(gdsc_grouped))
        self.assertTrue(all(c.startswith("EXPR_") for c in expr_cols))
        self.assertTrue(all(c.startswith("MUT_") for c in mut_cols))
        self.assertEqual(mgmt_cols, ["MGMT_Expr"])


if __name__ == "__main__":
    unittest.main()
