"""
Step 05: Post-hoc OOD error analysis, DepMap clinical subtype annotation,
Tanimoto scaffold novelty analysis, and publication figure generation.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mlp_workflow import load_saved_results_and_predictions
from src.results_analysis import run_results_analysis_workflow

if __name__ == "__main__":
    results, preds_store = load_saved_results_and_predictions()
    run_results_analysis_workflow(results, preds_store)
