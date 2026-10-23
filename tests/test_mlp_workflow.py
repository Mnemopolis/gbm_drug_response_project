"""Unit tests exclusively for `src.mlp_workflow`."""

import pickle
import tempfile
import unittest
from pathlib import Path

from src.mlp_workflow import load_saved_results_and_predictions


class TestMlpWorkflow(unittest.TestCase):
    """Verify loading and persistence of OOD results and prediction stores."""

    def test_load_saved_results_and_predictions_from_disk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            res_path = Path(tmp) / "OOD_results.pkl"
            preds_path = Path(tmp) / "OOD_preds.pkl"
            mock_res = {"random": {"test_rmse": 1.01, "test_spearman": 0.87}}
            mock_preds = {"random": {"preds": [1.0, 2.0], "trues": [1.1, 1.9]}}

            with open(res_path, "wb") as f1:
                pickle.dump(mock_res, f1)
            with open(preds_path, "wb") as f2:
                pickle.dump(mock_preds, f2)

            loaded_res, loaded_preds = load_saved_results_and_predictions(
                res_path, preds_path
            )
            self.assertEqual(loaded_res["random"]["test_spearman"], 0.87)
            self.assertEqual(len(loaded_preds["random"]["preds"]), 2)


if __name__ == "__main__":
    unittest.main()
