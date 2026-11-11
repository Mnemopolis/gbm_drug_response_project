"""
Unit tests for `src/workflow.py` (covering split, baseline, MLP, uncertainty, and master workflows).
"""

import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.workflow import (
    load_saved_results_and_predictions,
    run_baseline_workflow,
    run_master_workflow,
    run_split_workflow,
    run_uncertainty_workflow,
)
from tests.fixtures.synthetic_data import make_synthetic_pca_splits


class TestWorkflowOrchestrator(unittest.TestCase):
    """Verify all stage-level orchestrators and master CLI dispatch in `src/workflow.py`."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = make_synthetic_pca_splits(n_cells=8, n_drugs=10, n_pca=4, seed=42)

    def test_run_split_workflow_orchestrates_all_splits(self) -> None:
        raw_df = self.bundle["raw_df"]
        expr_cols = self.bundle["expr_cols"]
        mut_cols = self.bundle["mut_cols"]
        mgmt_cols = self.bundle["mgmt_cols"]

        with (
            patch("src.workflow.load_raw_gdsc_gbm_records", return_value=raw_df),
            patch("src.workflow.deduplicate_gdsc_drug_cell_pairs", return_value=raw_df),
            patch("src.workflow.extract_expression_features", return_value=(pd.DataFrame(), pd.DataFrame())),
            patch("src.workflow.extract_mutation_features", return_value=pd.DataFrame()),
            patch("src.workflow.build_drug_feature_table", return_value=pd.DataFrame()),
            patch(
                "src.workflow.merge_multimodal_dataset",
                return_value=(raw_df, expr_cols, mut_cols, mgmt_cols),
            ),
        ):
            splits = run_split_workflow(save_splits=False)
            self.assertEqual(set(splits.keys()), {"random", "leave_cell", "scaffold"})

    def test_run_baseline_workflow_returns_all_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            path_map = {}
            for split, (tr_df, te_df) in self.bundle["splits_dict"].items():
                tr_p, te_p = tmp / f"tr_{split}.pkl", tmp / f"te_{split}.pkl"
                tr_df.to_pickle(tr_p)
                te_df.to_pickle(te_p)
                path_map[split] = {"train": tr_p, "test": te_p}

            with patch("src.workflow.get_split_data_paths", side_effect=lambda s: path_map[s]):
                summary = run_baseline_workflow()
                self.assertEqual(set(summary.keys()), {"random", "leave_cell", "scaffold"})
                self.assertFalse(np.isnan(summary["scaffold"]["nearest_drug_spearman"]))

    def test_load_saved_results_and_predictions_from_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            res_p, pred_p = tmp / "res.pkl", tmp / "preds.pkl"
            with open(res_p, "wb") as f:
                pickle.dump(self.bundle["results_mock"], f)
            with open(pred_p, "wb") as f:
                pickle.dump(self.bundle["preds_store"], f)

            res, preds = load_saved_results_and_predictions(res_p, pred_p)
            self.assertIn("random", res)
            self.assertIn("scaffold", preds)

    def test_run_uncertainty_workflow_generates_all_tables(self) -> None:
        res_mock = {
            **self.bundle["results_mock"],
            "leave_cell": {
                **self.bundle["results_mock"]["leave_cell"],
                "seed_runs": [
                    {"test_preds": self.bundle["preds_store"]["leave_cell"]["preds"], "test_trues": self.bundle["preds_store"]["leave_cell"]["trues"]},
                    {"test_preds": self.bundle["preds_store"]["leave_cell"]["preds"], "test_trues": self.bundle["preds_store"]["leave_cell"]["trues"]},
                    {"test_preds": self.bundle["preds_store"]["leave_cell"]["preds"] * 0.1, "test_trues": self.bundle["preds_store"]["leave_cell"]["trues"]},
                ],
            },
        }
        out = run_uncertainty_workflow(res_mock, self.bundle["preds_store"])
        self.assertIn("seed_stability", out)
        self.assertIn("prediction_shrinkage", out)
        self.assertIn("leave_cell_seed_collapse", out)

    def test_run_master_workflow_dispatch(self) -> None:
        with (
            patch("src.workflow.run_split_workflow") as m_split,
            patch("src.workflow.run_baseline_workflow") as m_base,
            patch(
                "src.workflow.run_ood_mlp_workflow",
                return_value=(self.bundle["results_mock"], self.bundle["preds_store"]),
            ) as m_mlp,
            patch("src.workflow.run_uncertainty_workflow") as m_unc,
            patch("src.workflow.run_results_analysis_workflow") as m_ana,
        ):
            run_master_workflow(stage="data")
            m_split.assert_called_once()
            m_mlp.assert_not_called()

            run_master_workflow(stage="all", n_seeds=1, epochs=1)
            m_base.assert_called_once()
            m_mlp.assert_called_once()
            m_unc.assert_called_once()
            m_ana.assert_called_once()
