# Deep Learning for Glioblastoma (GBM) Drug Response Prediction & OOD Generalization

[![Healthcare AI Pipeline CI](https://github.com/your-username/gbm_drug_response/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/gbm_drug_response/actions)

**Course Final Mini-Project**: *Deep Learning in Healthcare AI*

## 1. Architectural Design Philosophy
Following industry-grade healthcare AI engineering practices, `src/` is organized into two distinct layers:
1. **Atomic Domain Modules (`*.py`)**: Pure, single-responsibility modules that each perform exactly one algorithmic or data-hygiene task (`acquisition.py`, `deduplication.py`, `fingerprints.py`, `splitting.py`, `standardization.py`, `pipeline_audit.py`, `eda.py`, `baselines.py`, `graph_data.py`, `mlp_gcn_model.py`, `mlp_gcn_training.py`, `uncertainty.py`, `results_analysis.py`).
2. **Stage Workflow Orchestrators (`*_workflow.py` & `workflow.py`)**: High-level pipeline coordinators (`split_workflow.py`, `baseline_workflow.py`, `mlp_gcn_workflow.py`, `uncertainty_workflow.py`, `workflow.py`) that connect atomic modules and manage file I/O.

---

## 2. Directory Tree

```text
gbm_drug_response/
├── .github/
│   └── workflows/
│       └── ci.yml                              # GitHub Actions CI: runs pytest & pipeline audit on every push
├── Data/                                       # Raw GDSC2, DepMap multi-omics, and SMILES datasets
├── Models/                                     # Cached splits (.pkl), model checkpoints (.pt), OOD results
├── Results/                                    # Publication-ready diagnostic figures (.png)
├── notebooks/                                  # Archive of original exploratory Jupyter Notebooks
├── src/                                        # Modular source code (Atomic Modules + Workflow Orchestrators)
│   ├── __init__.py
│   ├── config.py                               # Centralized pathlib paths, hyperparameters & GBM drug lists
│   ├── utils.py                                # Seed fixing, CUDA/MPS GPU device management & string helpers
│   ├── acquisition.py                          # Raw GDSC2, DepMap expression, mutation & methylation loading
│   ├── deduplication.py                        # Replicate IC50 averaging, SMILES deduplication & table merging
│   ├── fingerprints.py                         # RDKit 2048-bit Morgan fingerprints & Bemis-Murcko scaffolds
│   ├── splitting.py                            # Pure OOD splitting: Random, Leave-Cell-Out & Scaffold
│   ├── standardization.py                      # Train-only StandardScaler, PCA, MGMT median & Z-score scaling
│   ├── pipeline_audit.py                       # Automated anti-leakage & feature matrix integrity auditor
│   ├── eda.py                                  # Cohort exploratory statistics & split distribution histograms
│   ├── split_workflow.py                       # Stage 1 workflow: Acquisition -> Split -> PCA -> Audit
│   ├── baselines.py                            # Cell-only, Drug-only, Additive & Nearest-Drug FP baselines
│   ├── baseline_workflow.py                    # Stage 2 workflow: Statistical baseline decomposition
│   ├── graph_data.py                           # PyTorch Dataset, PyG Graph Dataset & GPU DataLoader
│   ├── mlp_gcn_model.py                        # Two-Tower BaselineMLP & GINConv GNNDrugMLP architectures
│   ├── mlp_gcn_training.py                     # Single-epoch GPU/CPU training, early stopping & evaluation
│   ├── mlp_gcn_workflow.py                     # Stage 3 workflow: Multi-split x multi-seed OOD training
│   ├── uncertainty.py                          # Multi-seed stability, Seed 2 collapse & variance shrinkage
│   ├── uncertainty_workflow.py                 # Stage 4 workflow: Uncertainty & shrinkage diagnostics
│   ├── results_analysis.py                     # Stage 5 workflow: Per-drug/cell errors, subtypes & 8 plots
│   └── workflow.py                             # Master end-to-end workflow orchestrator
├── scripts/                                    # Numbered step-by-step reproducible execution scripts
│   ├── 01_preprocess_and_split.py
│   ├── 02_evaluate_baselines.py
│   ├── 03_train_ood_models.py
│   ├── 04_uncertainty_and_shrinkage.py
│   ├── 05_error_analysis_and_plots.py
│   └── 06_validate_pipeline.py                 # Zero-CSV synthetic smoke test & anti-leakage audit (<3s)
├── tests/                                      # Automated unit test suite (pytest / unittest)
│   ├── __init__.py
│   ├── fixtures/
│   │   ├── __init__.py
│   │   └── synthetic_data.py
│   ├── test_config_and_deduplication.py
│   ├── test_splitting_standardization_audit.py
│   ├── test_baselines_and_models.py
│   └── test_uncertainty_and_results.py
├── main.py                                     # Unified CLI entry point
├── pyproject.toml                              # Modern Python PEP 621 project & pytest configuration
├── requirements.txt                            # Python dependencies
├── .gitignore                                  # Git ignore rules
└── README.md                                   # Project documentation
```

---

## 3. Quick Start in `.vene`

```bash
source .vene/bin/activate        # Windows: .vene\Scripts\activate
pip install -r requirements.txt
pip install -e .                 # Optional: install package in editable mode via pyproject.toml
```

### Run Automated Tests & Anti-Leakage Audit
```bash
pytest -v
python scripts/06_validate_pipeline.py
```

### Run Full Pipeline (or Step-by-Step Scripts)
```bash
# Unified CLI:
python main.py --stage all --device auto

# Or step-by-step via scripts/:
python scripts/01_preprocess_and_split.py
python scripts/02_evaluate_baselines.py
python scripts/03_train_ood_models.py --device auto
python scripts/04_uncertainty_and_shrinkage.py
python scripts/05_error_analysis_and_plots.py
```
