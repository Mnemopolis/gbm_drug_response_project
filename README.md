# Deep Learning for Glioblastoma (GBM) Drug Response & OOD Generalization

**Course Final Mini-Project**: *Deep Learning in Healthcare AI*

---

## 1. Project Overview

Predicting anticancer drug sensitivity (`LN_IC50`) from multimodal biological and chemical data is a central problem in precision oncology. However, standard **in-distribution random splits** often overestimate clinical utility due to **shortcut learning** (memorizing average drug toxicity rather than learning patient-specific drug–gene interactions).

This repository implements a leakage-free multimodal deep learning benchmark for **Glioblastoma (GBM)** drug response prediction combining **GDSC2** dose-response records, **DepMap** multi-omics profiles (gene expression, hotspot mutations, and `MGMT` status), and **RDKit** molecular structures ($5,107$ unique `(drug, cell_line)` pairs across $24$ GBM cell lines and $243$ drugs). We evaluate a **Two-Tower `BaselineMLP`** against statistical baselines across three regimes:
1. **Random Split (In-Distribution)**: Random `(drug, cell_line)` pair holdout ($80\%$ train / $20\%$ test).
2. **Leave-Cell-Out Split (Biological OOD)**: Strict holdout of unseen GBM cell lines (`ModelID`).
3. **Bemis–Murcko Scaffold Split (Chemical OOD)**: Strict holdout of unseen chemical scaffolds.

---

## 2. Repository Structure

```text
gbm_drug_response_project/
├── .github/workflows/ci.yml              # GitHub Actions CI & Lectrace Pages deployment
├── Data/                                 # Raw GDSC2, DepMap & SMILES CSVs (Git-ignored)
├── Lectrace/                             # Interactive Execution-Explanation Presentation
│   ├── _project_data.py                  # Small, reproducible GBM inputs & benchmarks
│   └── 01_gbm_ood_presentation.py        # 10-minute Lectrace presentation script
├── Models/                               # Saved splits (.pkl) & checkpoints (.pt) (Git-ignored)
├── Notebooks/                            # Original exploratory Jupyter Notebooks (01 - 04)
├── Results/                              # Generated diagnostic figures (.png)
├── src/                                  # Modular Python package (Single Responsibility Principle)
│   ├── config.py, utils.py               # Paths, hyperparameters, random seeds & GPU device setup
│   ├── acquisition.py, deduplication.py  # Raw data ingestion, replicate averaging & alignment
│   ├── fingerprints.py                   # RDKit Morgan fingerprints (2048-bit) & Murcko scaffolds
│   ├── splitting.py, standardization.py  # OOD splitting & train-only PCA / Z-score scaling
│   ├── pipeline_audit.py, eda.py         # Anti-leakage auditor & split distribution plots
│   ├── baselines.py, dataset.py          # Statistical baselines & PyTorch Dataset/DataLoader
│   ├── mlp_model.py, mlp_training.py     # Two-Tower BaselineMLP & single-epoch training/eval
│   ├── uncertainty.py, results_analysis.py # Variance shrinkage & error analysis plots
│   └── *_workflow.py, workflow.py        # Stage orchestrators & master pipeline workflow
├── scripts/                              # Step-by-step CLI scripts (01 to 06)
├── tests/                                # 1-to-1 unit test suite (21 test files, incl. test_lectrace.py)
├── main.py                               # Unified CLI entry point
├── pyproject.toml & requirements.txt     # Package metadata & dependencies
└── README.md
```

---

## 3. Quick Start

### 3.1 Environment Setup
```bash
source .vene/bin/activate        # Windows: .vene\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3.2 Run Unit Tests & Synthetic Pipeline Audit (No Raw CSVs Needed)
```bash
pytest -v
python scripts/06_validate_pipeline.py
```

### 3.3 Run Full Training & Evaluation Pipeline
Place raw CSV files (`GDSC2_fitted_dose_response_27Oct23.csv`, `Model.csv`, `OmicsExpressionProteinCodingGenesTPMLogp1BatchCorrected.csv`, `OmicsSomaticMutationsMatrixHotspot.csv`, `CCLE_RRBS_TSS_1kb_20180614.txt`, `drug_smiles.csv`) in `Data/`, then run:
```bash
# Run all stages via unified entry point:
python main.py --stage all --device auto

# Or run stage-by-stage:
python scripts/01_preprocess_and_split.py
python scripts/02_evaluate_baselines.py
python scripts/03_train_ood_models.py --seeds 5 --epochs 100 --device auto
python scripts/04_uncertainty_and_shrinkage.py
python scripts/05_error_analysis_and_plots.py
```

---

## 4. Execution-Explanation Presentation Using Lectrace (`Lectrace/`)

The `Lectrace/` directory contains two files for our 10-minute interactive execution-explanation presentation:
- `Lectrace/_project_data.py`: Small, reproducible GBM cell-line, drug, and benchmark inputs implementing `__lectrace__()`.
- `Lectrace/01_gbm_ood_presentation.py`: Main Lectrace script with narrative `text()`, speaker `note()`, `Reference()` code links, `table()`, and Vega-Lite `plot()`.

### 4.1 How to Run Lectrace (Official Lectrace CLI)
Install Lectrace (`pip install git+https://github.com/CSC3160/lectrace.git`) and launch the local interactive server from the repository root:
```bash
lectrace serve Lectrace/01_gbm_ood_presentation.py --workspace-root .
```
To build the static HTML bundle for **GitHub Pages** deployment:
```bash
lectrace build Lectrace/01_gbm_ood_presentation.py --workspace-root . -o _site
```

### 4.2 Selected Execution States (10-Minute Walkthrough)

- **Introduction (Minutes 0:00 – 2:30)**: Introduces GBM clinical resistance to Temozolomide (`MGMT` repair), the danger of shortcut learning on random splits, and our Two-Tower `BaselineMLP` architecture ($5,107$ GBM drug–cell pairs).
- **Execution State 1 — Leakage-Free OOD Splitting & Train-Only Standardization (`trace_state_1_leakage_free_preprocessing`)**:
  - *What happens*: Partitions cell lines (`Leave-Cell-Out`) and Bemis–Murcko scaffolds (`Scaffold`), fitting gene-expression mean/std strictly on training cell lines.
  - *What changes*: `cell_overlap` and `scaffold_overlap` evaluate to `0` (`leakage_audit_passed = True`), and held-out `GAMG` is standardized using frozen training moments (`gamg_z_gene0_standardized = +3.5769`).
  - *Why it matters*: Prevents test-set gene covariance and drug potency priors from leaking into training representations.
- **Execution State 2 — Two-Tower `BaselineMLP` Forward Pass & Latent Fusion (`trace_state_2_two_tower_forward_pass`)**:
  - *What happens*: Traces a forward pass for **Temozolomide** on `MGMT`-low **U-87 MG** (`mgmt_expr = 0.42`, sensitive) vs. `MGMT`-high **T98G** (`mgmt_expr = 5.84`, resistant).
  - *What changes*: Concatenating drug (`drug_emb_4d`) and cell embeddings shifts predicted `LN_IC50` from `5.663` (`U-87 MG`) to `6.921` (`T98G`), yielding `mgmt_resistance_shift_ln_ic50 = +1.258`.
  - *Why it matters*: Demonstrates how late fusion captures `MGMT`-mediated relative resistance between patients treated with the same drug.
- **Execution State 3 — Shortcut Learning Diagnosis via Additive Baseline Decomposition (`trace_state_3_shortcut_decomposition`)**:
  - *What happens*: Decomposes Random-split predictions into `Cell-Only` ($0.2107$), `Drug-Only` ($0.8349$), and interaction-free `Additive` ($\bar{y}_d + \bar{y}_c - \bar{y}_{\text{global}} = 0.8752$) baselines vs. `BaselineMLP` ($0.8744$).
  - *What changes*: Nonlinear interaction gain evaluates to `interaction_gain_global = -0.0008` and `interaction_gain_within_cell = -0.0013`, with `drug_potency_share_pct = 95.5%`.
  - *Why it matters*: Proves that high in-distribution Spearman is driven by memorizing per-drug average toxicity rather than learning complex drug–gene interactions.
- **Execution State 4 — Chemical OOD Collapse, Variance Shrinkage & Decoupling (`trace_state_4_ood_shrinkage_and_decoupling`)**:
  - *What happens*: Evaluates `BaselineMLP` on $47$ unseen drug scaffolds (`Scaffold` split) and compares global metrics against within-drug cell-line rankings.
  - *What changes*: Global Spearman drops from $0.8744 \to 0.2905$ as prediction variance shrinks ($\sigma(\hat{y})/\sigma(y) = 0.663$, under-predicting Temozolomide by $-3.13$ log units), **yet median within-drug Spearman across cell lines remains $0.530$** (vs. $0.500$ on Random and Leave-Cell-Out).
  - *Why it matters*: Reveals that the cell tower still ranks relative patient susceptibility for novel drugs, whereas the OOD bottleneck is calibrating the absolute dose offset of unseen chemical scaffolds.
- **Conclusion (Minutes 9:00 – 10:00)**: Summarizes why clinical oncology AI must pair OOD splits with additive baselines, using multi-omics models for relative patient stratification rather than uncalibrated dosing on novel compounds.

---

## 5. Key Experimental Results

### 5.1 Multi-Split Benchmark Summary ($5$ Random Seeds)

| Split Regime | Cell-Only | Drug-Only | Additive (No Inter.) | Nearest-Drug FP | `BaselineMLP` Global Spearman | `BaselineMLP` Global RMSE | Within-Cell `BaselineMLP` | Within-Cell Drug-Only |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random (In-Dist)** | $0.2107$ | $0.8349$ | $0.8752$ | — | **$0.8744 \pm 0.0051$** | **$1.0346 \pm 0.0211$** | $0.8541 \pm 0.0576$ | $0.8554 \pm 0.0598$ |
| **Leave-Cell-Out (Bio OOD)** | N/A | $0.8362$ | N/A | — | $0.6947 \pm 0.2979^*$ | $1.3970 \pm 0.6115^*$ | **$0.8460 \pm 0.0608$** | $0.8449 \pm 0.0602$ |
| **Scaffold (Chem OOD)** | $0.2494$ | N/A | N/A | $0.1445$ | **$0.2905 \pm 0.0045$** | $2.1951 \pm 0.0081$ | $0.1735 \pm 0.0813$ | N/A |

*\*In Leave-Cell-Out, 4 of 5 seeds converge to Spearman $0.8388\text{–}0.8476$ (mean $0.8436$), while `seed=2` early-stops at epoch 21 ($0.0989$).*

### 5.2 Diagnostic Figures (`Results/`)
- **Baseline & OOD Comparison**: `Results/diag_spearman_decomposition.png`, `Results/OOD_three_split_comparison.png`
- **Variance Shrinkage & Bias**: `Results/OOD_scatter_three_splits.png`, `Results/OOD_prediction_bias_box.png`
- **Global vs. Within-Drug Decoupling**: `Results/OOD_global_vs_within_drug_spearman.png`, `Results/OOD_per_drug_rmse_box.png`
- **Scaffold Novelty vs. Error**: `Results/OOD_scaffold_novelty_vs_error.png`

---

## 6. License

Released under the [MIT License](LICENSE).
