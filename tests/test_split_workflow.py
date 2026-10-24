"""Unit tests exclusively for `src.split_workflow`."""

import unittest
from unittest.mock import patch

from src.split_workflow import run_split_workflow
from tests.fixtures.synthetic_data import make_synthetic_merged_dataframe


class TestSplitWorkflow(unittest.TestCase):
    """Verify Stage 1 split workflow orchestration using mocked raw data loaders."""

    def test_run_split_workflow_orchestrates_all_splits(self) -> None:
        synth_df = make_synthetic_merged_dataframe(n_cells=8, n_drugs=8, seed=42)
        expr_cols = [c for c in synth_df.columns if c.startswith("EXPR_")]
        mut_cols = [c for c in synth_df.columns if c.startswith("MUT_")]

        expr_gbm = (
            synth_df[["ModelID"] + expr_cols]
            .drop_duplicates("ModelID")
            .set_index("ModelID")
        )
        expr_gbm.columns = [c.replace("EXPR_", "") for c in expr_cols]
        mut_gbm = (
            synth_df[["ModelID"] + mut_cols]
            .drop_duplicates("ModelID")
            .set_index("ModelID")
        )
        mut_gbm.columns = [c.replace("MUT_", "") for c in mut_cols]
        mgmt_expr = (
            synth_df[["ModelID", "MGMT_Expr"]]
            .drop_duplicates("ModelID")
            .set_index("ModelID")
        )
        drug_info = synth_df[
            ["DRUG_NAME", "SMILES", "DRUG_KEY", "Morgan_FP", "Scaffold"]
        ].drop_duplicates("DRUG_KEY")

        with (
            patch(
                "src.split_workflow.load_raw_gdsc_gbm_records", return_value=synth_df
            ),
            patch(
                "src.split_workflow.extract_expression_features",
                return_value=(expr_gbm, mgmt_expr),
            ),
            patch("src.split_workflow.extract_mutation_features", return_value=mut_gbm),
            patch(
                "src.split_workflow.convert_and_extract_methylation_features",
                return_value=None,
            ),
            patch(
                "src.split_workflow.build_drug_feature_table", return_value=drug_info
            ),
        ):
            splits_dict = run_split_workflow(save_splits=False)

        self.assertEqual(set(splits_dict.keys()), {"random", "leave_cell", "scaffold"})
        for split_name, (tr_df, te_df) in splits_dict.items():
            self.assertGreater(len(tr_df), 0, msg=f"Empty train in {split_name}")
            self.assertGreater(len(te_df), 0, msg=f"Empty test in {split_name}")


if __name__ == "__main__":
    unittest.main()
