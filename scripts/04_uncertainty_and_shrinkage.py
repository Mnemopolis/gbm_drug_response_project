"""
Step 04: Run multi-seed training stability audit, Leave-Cell-Out Seed 2 collapse
diagnosis, and OOD prediction variance shrinkage analysis.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mlp_workflow import load_saved_results_and_predictions
from src.uncertainty_workflow import run_uncertainty_workflow

if __name__ == "__main__":
    results, preds_store = load_saved_results_and_predictions()
    run_uncertainty_workflow(results, preds_store)
