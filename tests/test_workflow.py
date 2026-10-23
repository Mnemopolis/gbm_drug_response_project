"""Unit tests exclusively for `src.workflow`."""

import unittest
from unittest.mock import patch
from src.workflow import run_master_workflow


class TestWorkflow(unittest.TestCase):
    """Verify master workflow routing for individual and full pipeline stages."""

    def test_run_master_workflow_data_stage_only(self) -> None:
        with patch("src.workflow.run_split_workflow") as mock_split,              patch("src.workflow.run_baseline_workflow") as mock_base:
            run_master_workflow(stage="data")
            mock_split.assert_called_once_with(save_splits=True)
            mock_base.assert_not_called()

    def test_run_master_workflow_all_stages(self) -> None:
        with patch("src.workflow.run_split_workflow") as mock_split,              patch("src.workflow.run_baseline_workflow") as mock_base,              patch("src.workflow.run_ood_mlp_workflow", return_value=({}, {})) as mock_train,              patch("src.workflow.run_uncertainty_workflow") as mock_unc,              patch("src.workflow.run_results_analysis_workflow") as mock_res:
            run_master_workflow(stage="all", n_seeds=2, epochs=5, patience=2)
            mock_split.assert_called_once()
            mock_base.assert_called_once()
            mock_train.assert_called_once()
            mock_unc.assert_called_once()
            mock_res.assert_called_once()


if __name__ == "__main__":
    unittest.main()
