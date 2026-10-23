"""Unit tests exclusively for `src.uncertainty_workflow`."""

import unittest
import numpy as np
import pandas as pd

from src.uncertainty_workflow import run_uncertainty_workflow


class TestUncertaintyWorkflow(unittest.TestCase):
    """Verify Stage 4 uncertainty and stability workflow orchestrator."""

    def test_run_uncertainty_workflow_generates_all_tables(self) -> None:
        trues = np.array([1.0, 2.0, 3.0, 4.0])
        preds0 = np.array([1.1, 2.0, 2.9, 4.1])
        preds2 = np.array([2.5, 2.6, 2.5, 2.7])
        test_df = pd.DataFrame({"ModelID": ["C1", "C1", "C1", "C1"], "LN_IC50": trues})

        results = {
            "leave_cell": {
                "model_global": {"values": [0.85, 0.84, 0.10]},
                "model_rmse": {"mean": 1.4, "values": [1.0, 1.1, 2.6]},
                "seed_runs": [
                    {"test_preds": preds0, "test_trues": trues},
                    {"test_preds": preds0, "test_trues": trues},
                    {"test_preds": preds2, "test_trues": trues},
                ],
            }
        }
        preds_store = {
            "leave_cell": {"test_df": test_df, "preds": preds0, "trues": trues}
        }

        outputs = run_uncertainty_workflow(results, preds_store)
        self.assertIn("seed_stability", outputs)
        self.assertIn("prediction_shrinkage", outputs)
        self.assertIn("leave_cell_seed_collapse", outputs)


if __name__ == "__main__":
    unittest.main()
