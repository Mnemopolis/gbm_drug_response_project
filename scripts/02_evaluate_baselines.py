"""
Step 02: Evaluate statistical baselines (Cell-only, Drug-only, Additive) and
chemical similarity (Nearest-Drug Morgan FP) baselines across splits.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.baseline_workflow import run_baseline_workflow

if __name__ == "__main__":
    run_baseline_workflow()
