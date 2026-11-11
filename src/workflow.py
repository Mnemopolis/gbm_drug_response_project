"""
Unified Pipeline Workflow Orchestrator (`src/workflow.py`).

Consolidates all stage-level workflows:
1. `run_split_workflow`: Multi-omics acquisition, deduplication, RDKit fingerprints,
   OOD splitting, train-only PCA standardization, and anti-leakage audit.
2. `run_baseline_workflow`: Statistical (Cell-only, Drug-only, Additive) and
   Nearest-Drug Morgan FP chemical similarity baselines across splits.
3. `run_ood_mlp_workflow` & `run_random_mlp_workflow`: Multi-split x multi-seed
   Two-Tower `BaselineMLP` training and evaluation.
4. `run_uncertainty_workflow`: Multi-seed stability, Seed 2 collapse diagnosis,
   and OOD prediction variance shrinkage analysis.
5. `run_master_workflow`: End-to-end CLI entrypoint.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.acquisition import (
    extract_expression_features,
    extract_mutation_features,
    load_raw_gdsc_gbm_records,
)
from src.baselines import (
    compute_drug_only_predictions,
    compute_within_cell_spearman,
    evaluate_all_statistical_baselines,
    evaluate_nearest_drug_scaffold_baseline,
)
from src.config import (
    BEST_RANDOM_MODEL_PATH,
    DEFAULT_SEED,
    EARLY_STOPPING_PATIENCE,
    EXPR_FILE,
    GDSC_FILE,
    MAX_EPOCHS,
    MODEL_FILE,
    MUT_FILE,
    N_MODEL_SEEDS,
    OOD_PREDS_PKL,
    OOD_RESULTS_PKL,
    SMILES_FILE,
    SPLIT_NAMES,
    get_split_data_paths,
)
from src.deduplication import deduplicate_gdsc_drug_cell_pairs, merge_multimodal_dataset
from src.eda import compute_cohort_eda_summary, plot_label_distribution_across_splits
from src.fingerprints import build_drug_feature_table
from src.mlp_training import train_single_seed
from src.pipeline_audit import run_full_pipeline_audit
from src.results_analysis import run_results_analysis_workflow
from src.splitting import partition_dataframe_by_split, partition_train_validation
from src.standardization import fit_transform_cell_pca_features
from src.uncertainty import (
    compare_cell_performance_between_seeds,
    summarize_prediction_shrinkage_and_bias,
    summarize_seed_stability_across_splits,
)
from src.utils import get_device, log_device_info


def run_split_workflow(
    save_splits: bool = True,
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Stage 1: Acquire raw multi-omics, deduplicate, featurize drugs, split, fit PCA, and audit."""
    print("=" * 70)
    print("WORKFLOW 1: Multi-Omics Acquisition, Deduplication & Leakage-Free Splitting")
    print("=" * 70)

    raw_gbm = load_raw_gdsc_gbm_records(GDSC_FILE, MODEL_FILE)
    gdsc_grouped = deduplicate_gdsc_drug_cell_pairs(raw_gbm)
    valid_cells = gdsc_grouped["ModelID"].unique()

    expr_gbm, mgmt_expr = extract_expression_features(EXPR_FILE, valid_cells)
    mut_gbm = extract_mutation_features(MUT_FILE, valid_cells)
    drug_info = build_drug_feature_table(SMILES_FILE)

    final_df, expr_cols, mut_cols, mgmt_cols = merge_multimodal_dataset(
        gdsc_grouped, drug_info, expr_gbm, mut_gbm, mgmt_expr
    )
    eda = compute_cohort_eda_summary(final_df)
    print(
        f"[Cohort] Pairs={eda['n_records']} | Cells={eda['n_unique_cells']} | "
        f"Drugs={eda['n_unique_drugs']} | Scaffolds={eda['n_unique_scaffolds']} | "
        f"LN_IC50={eda['ic50_mean']:.2f}+/-{eda['ic50_std']:.2f}"
    )

    splits_dict: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for split_name in SPLIT_NAMES:
        raw_train, raw_test = partition_dataframe_by_split(final_df, split_name)
        train_final, test_final = fit_transform_cell_pca_features(
            raw_train, raw_test, expr_cols, mut_cols, mgmt_cols
        )
        splits_dict[split_name] = (train_final, test_final)
        print(f"  [{split_name:<10s}] train={train_final.shape} | test={test_final.shape}")
        if save_splits:
            paths = get_split_data_paths(split_name)
            train_final.to_pickle(paths["train"])
            test_final.to_pickle(paths["test"])

    audit_df = run_full_pipeline_audit(splits_dict)
    print("\n[Pipeline Audit] Anti-Leakage Verification Passed:")
    print(audit_df.to_string(index=False))

    if save_splits:
        plot_label_distribution_across_splits(splits_dict)

    return splits_dict


def run_baseline_workflow() -> dict[str, dict[str, Any]]:
    """Stage 2: Evaluate non-deep-learning statistical and chemical similarity baselines."""
    print("=" * 70)
    print("WORKFLOW 2: Evaluating Statistical & Chemical Similarity Baselines")
    print("=" * 70)

    summary: dict[str, dict[str, Any]] = {}
    for split in SPLIT_NAMES:
        paths = get_split_data_paths(split)
        trainval_df = pd.read_pickle(paths["train"])
        test_df = pd.read_pickle(paths["test"]).reset_index(drop=True)
        train_df, _ = partition_train_validation(trainval_df)

        stats = evaluate_all_statistical_baselines(train_df, test_df)
        pred_drug = compute_drug_only_predictions(train_df, test_df)
        mask = pred_drug.notna()
        wc_mean, wc_std, wc_n = compute_within_cell_spearman(
            pred_drug[mask].values,
            test_df.loc[mask, "LN_IC50"].values,
            test_df.loc[mask, "ModelID"].values,
        )
        nn_sp = (
            evaluate_nearest_drug_scaffold_baseline(trainval_df, test_df)
            if split == "scaffold"
            else float("nan")
        )
        summary[split] = {
            **stats,
            "within_cell_drug_only_mean": wc_mean,
            "within_cell_drug_only_std": wc_std,
            "within_cell_drug_only_n": wc_n,
            "nearest_drug_spearman": nn_sp,
        }
        print(
            f"  [{split:<10s}] Cell={stats['cell_only']:.4f} | Drug={stats['drug_only']:.4f} | "
            f"Additive={stats['additive']:.4f} | WC-Drug={wc_mean:.4f} | Nearest-Drug={nn_sp:.4f}"
        )
    return summary


def run_ood_mlp_workflow(
    split_names: list[str] = SPLIT_NAMES,
    n_seeds: int = N_MODEL_SEEDS,
    epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
    prefer_device: str = "auto",
    require_gpu: bool = False,
    save_results: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Stage 3: Train and evaluate `BaselineMLP` across splits and seeds."""
    device = get_device(prefer_device=prefer_device, require_gpu=require_gpu)
    print("=" * 70)
    print(f"WORKFLOW 3: Multi-Split OOD Experiments ({len(split_names)} splits x {n_seeds} seeds)")
    log_device_info(device)
    print("=" * 70)

    results: dict[str, Any] = {}
    preds_store: dict[str, Any] = {}

    for split in split_names:
        paths = get_split_data_paths(split)
        trainval_df = pd.read_pickle(paths["train"])
        test_df = pd.read_pickle(paths["test"]).reset_index(drop=True)
        train_df, val_df = partition_train_validation(trainval_df)
        baselines = evaluate_all_statistical_baselines(train_df, test_df)

        seed_runs: list[dict[str, Any]] = []
        for seed in range(n_seeds):
            ckpt_path = BEST_RANDOM_MODEL_PATH if (split == "random" and seed == 0) else None
            run_metrics = train_single_seed(
                train_df,
                val_df,
                test_df,
                seed=seed,
                epochs=epochs,
                patience=patience,
                device=device,
                save_checkpoint_path=ckpt_path,
            )
            seed_runs.append(run_metrics)
            print(
                f"  [{split} | seed={seed}] RMSE={run_metrics['test_rmse']:.4f} | "
                f"Spearman={run_metrics['test_spearman']:.4f} | epochs={run_metrics['n_epochs']}"
            )

        seed0 = seed_runs[0]
        wc_mean, wc_std, wc_n = compute_within_cell_spearman(
            seed0["test_preds"], seed0["test_trues"], test_df["ModelID"].values
        )
        pred_drug = compute_drug_only_predictions(train_df, test_df)
        mask = pred_drug.notna()
        wc_drug_mean, wc_drug_std, wc_drug_n = compute_within_cell_spearman(
            pred_drug[mask].values,
            test_df.loc[mask, "LN_IC50"].values,
            test_df.loc[mask, "ModelID"].values,
        )

        sp_vals = [float(r["test_spearman"]) for r in seed_runs]
        rmse_vals = [float(r["test_rmse"]) for r in seed_runs]
        results[split] = {
            "baselines": baselines,
            "model_global": {"mean": float(np.mean(sp_vals)), "std": float(np.std(sp_vals)), "values": sp_vals},
            "model_rmse": {"mean": float(np.mean(rmse_vals)), "std": float(np.std(rmse_vals)), "values": rmse_vals},
            "model_within_cell": {"mean": wc_mean, "std": wc_std, "n_cells": wc_n},
            "within_cell_drug_only": {"mean": wc_drug_mean, "std": wc_drug_std, "n_cells": wc_drug_n},
            "seed_runs": seed_runs,
            "test_preds": seed0["test_preds"],
            "test_trues": seed0["test_trues"],
            "test_spearman": seed0["test_spearman"],
            "test_rmse": seed0["test_rmse"],
        }
        preds_store[split] = {
            "test_df": test_df,
            "preds": seed0["test_preds"],
            "trues": seed0["test_trues"],
            "spearman": seed0["test_spearman"],
            "rmse": seed0["test_rmse"],
        }

    if save_results:
        with open(OOD_RESULTS_PKL, "wb") as f_res:
            pickle.dump(results, f_res)
        with open(OOD_PREDS_PKL, "wb") as f_pred:
            pickle.dump(preds_store, f_pred)

    return results, preds_store


def run_random_mlp_workflow(
    seed: int = DEFAULT_SEED,
    epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
    prefer_device: str = "auto",
    require_gpu: bool = False,
) -> dict[str, Any]:
    """Single-split convenience wrapper delegating to `run_ood_mlp_workflow` on `random`."""
    results, _ = run_ood_mlp_workflow(
        split_names=["random"],
        n_seeds=1,
        epochs=epochs,
        patience=patience,
        prefer_device=prefer_device,
        require_gpu=require_gpu,
        save_results=False,
    )
    return results["random"]


def load_saved_results_and_predictions(
    ood_results_path: Path = OOD_RESULTS_PKL,
    ood_preds_path: Path = OOD_PREDS_PKL,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load pre-computed `OOD_results.pkl` and `OOD_preds.pkl` from `Models/`."""
    if not ood_results_path.exists():
        raise FileNotFoundError(f"Missing {ood_results_path}. Run training first.")

    with open(ood_results_path, "rb") as f_res:
        results = pickle.load(f_res)

    if ood_preds_path.exists():
        with open(ood_preds_path, "rb") as f_pred:
            preds_store = pickle.load(f_pred)
    else:
        preds_store = {
            split: {
                "test_df": pd.read_pickle(get_split_data_paths(split)["test"]).reset_index(drop=True),
                "preds": results[split]["test_preds"],
                "trues": results[split]["test_trues"],
                "spearman": results[split]["test_spearman"],
                "rmse": results[split]["test_rmse"],
            }
            for split in SPLIT_NAMES
        }
        with open(ood_preds_path, "wb") as f_pred:
            pickle.dump(preds_store, f_pred)

    return results, preds_store


def run_uncertainty_workflow(
    results: dict[str, Any],
    preds_store: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    """Stage 4: Run multi-seed stability and prediction shrinkage diagnostics across splits."""
    print("\n" + "=" * 70)
    print("WORKFLOW 3B: Multi-Seed Uncertainty & Prediction Shrinkage Diagnostics")
    print("=" * 70)

    stability_df = summarize_seed_stability_across_splits(results)
    shrinkage_df = summarize_prediction_shrinkage_and_bias(preds_store)
    print("\n--- Multi-Seed Training Stability Across Splits ---")
    print(stability_df.to_string(index=False))
    print("\n--- Prediction Variance Shrinkage (pred_std / true_std) & Bias ---")
    print(shrinkage_df.to_string(index=False))

    outputs: dict[str, pd.DataFrame] = {
        "seed_stability": stability_df,
        "prediction_shrinkage": shrinkage_df,
    }
    if "leave_cell" in results and len(results["leave_cell"].get("seed_runs", [])) >= 3:
        seed_runs = results["leave_cell"]["seed_runs"]
        collapse_df = compare_cell_performance_between_seeds(
            preds_seed_a=seed_runs[0]["test_preds"],
            preds_seed_b=seed_runs[2]["test_preds"],
            trues=seed_runs[0]["test_trues"],
            cell_ids=preds_store["leave_cell"]["test_df"]["ModelID"].values,
        )
        print("\n--- Leave-Cell-Out Seed 0 vs. Seed 2 Per-Cell Collapse Diagnosis ---")
        print(collapse_df.to_string(index=False))
        outputs["leave_cell_seed_collapse"] = collapse_df

    return outputs


def run_master_workflow(
    stage: str = "all",
    n_seeds: int = N_MODEL_SEEDS,
    epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
    prefer_device: str = "auto",
    require_gpu: bool = False,
) -> None:
    """Execute the requested pipeline stage(s) from start to finish."""
    if stage in ("all", "data"):
        run_split_workflow(save_splits=True)

    if stage in ("all", "baselines"):
        run_baseline_workflow()

    if stage in ("all", "train"):
        results, preds_store = run_ood_mlp_workflow(
            split_names=SPLIT_NAMES,
            n_seeds=n_seeds,
            epochs=epochs,
            patience=patience,
            prefer_device=prefer_device,
            require_gpu=require_gpu,
            save_results=True,
        )
    elif stage in ("uncertainty", "analyze"):
        results, preds_store = load_saved_results_and_predictions()
    else:
        return

    if stage in ("all", "train", "uncertainty", "analyze"):
        run_uncertainty_workflow(results, preds_store)

    if stage in ("all", "train", "analyze"):
        run_results_analysis_workflow(results, preds_store)
