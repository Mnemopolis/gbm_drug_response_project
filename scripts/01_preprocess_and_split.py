"""
Step 01: Multi-omics acquisition, deduplication, RDKit fingerprints, OOD splitting,
leakage-free PCA standardization, and pipeline audit.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.split_workflow import run_split_workflow

if __name__ == "__main__":
    run_split_workflow(save_splits=True)
