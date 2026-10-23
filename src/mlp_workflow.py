"""
Deep Learning Training & Multi-Split OOD Benchmark Workflow Orchestrator.

Single Responsibility Principle (SRP):
Orchestrates single-split training (Notebook 02) and 3-split x 5-seed OOD
benchmarking (Notebook 03), persisting `OOD_results.pkl` and `OOD_preds.pkl`.
"""

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.baselines import (
    compute_drug_only_predictions,
    compute_within_cell_spearman,
    evaluate_all_statistical_baselines,
)
from src.config import (
    BEST_RANDOM_MODEL_PATH,
    DEFAULT_SEED,
    EARLY_STOPPING_PATIENCE,
    MAX_EPOCHS,
    N_MODEL_SEEDS,
    OOD_PREDS_PKL,
    OOD_RESULTS_PKL,
    SPLIT_NAMES,
    get_split_data_paths,
)
from src.mlp_training import train_single_seed
from src.splitting import partition_train_validation
from src.utils import get_device, log_device_info


def run_random_mlp_workflow(
    seed: int = DEFAULT_SEED,
    epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
    prefer_device: str = "auto",
    require_gpu: bool = False,
) -> dict[str, Any]:
    """Train BaselineMLP on the random split and return model + baseline decomposition metrics."""
    device = get_device(prefer_device=prefer_device, require_gpu=require_gpu)
    log_device_info(device)

    paths = get_split_data_paths("random")
    trainval_df = pd.read_pickle(paths["train"])
    test_df = pd.read_pickle(paths["test"]).reset_index(drop=True)
    train_df, val_df = partition_train_validation(trainval_df, seed=seed)

    run_metrics = train_single_seed(
        train_df,
        val_df,
        test_df,
        seed=seed,
        epochs=epochs,
        patience=patience,
        device=device,
        save_checkpoint_path=BEST_RANDOM_MODEL_PATH,
        verbose=True,
    )

    baselines = evaluate_all_statistical_baselines(train_df, test_df)
    wc_mean, wc_std, wc_n = compute_within_cell_spearman(
        run_metrics["test_preds"],
        run_metrics["test_trues"],
        test_df["ModelID"].values,
        min_n=20,
    )

    return {
        "baselines": baselines,
        "model_global": {"mean": run_metrics["test_spearman"], "std": 0.0},
        "model_within_cell": {"mean": wc_mean, "std": wc_std, "n_cells": wc_n},
        "test_rmse": run_metrics["test_rmse"],
        "test_spearman": run_metrics["test_spearman"],
    }


def run_ood_mlp_workflow(
    split_names: list[str] = SPLIT_NAMES,
    n_seeds: int = N_MODEL_SEEDS,
    epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
    prefer_device: str = "auto",
    require_gpu: bool = False,
    save_results: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Execute the multi-split x multi-seed OOD evaluation benchmark on GPU (or fallback CPU),
    save `OOD_results.pkl` and `OOD_preds.pkl`, and return both dictionaries.
    """
    device = get_device(prefer_device=prefer_device, require_gpu=require_gpu)
    print("=" * 70)
    print(
        f"WORKFLOW 2: Running Multi-Split OOD Experiments ({len(split_names)} splits x {n_seeds} seeds)"
    )
    log_device_info(device)
    print("=" * 70)

    results: dict[str, Any] = {}
    preds_store: dict[str, Any] = {}

    for split in split_names:
        print(f"\n{'=' * 60}\nSplit: {split}\n{'=' * 60}")
        paths = get_split_data_paths(split)
        trainval_df = pd.read_pickle(paths["train"])
        test_df = pd.read_pickle(paths["test"]).reset_index(drop=True)

        train_df, val_df = partition_train_validation(trainval_df)
        baselines = evaluate_all_statistical_baselines(train_df, test_df)

        seed_runs: list[dict[str, Any]] = []
        for seed in range(n_seeds):
            ckpt_path = (
                BEST_RANDOM_MODEL_PATH if (split == "random" and seed == 0) else None
            )
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
                f"  [seed={seed}] RMSE={run_metrics['test_rmse']:.4f} | "
                f"Spearman={run_metrics['test_spearman']:.4f} | "
                f"epochs={run_metrics['n_epochs']}"
            )

        seed0 = seed_runs[0]
        wc_mean, wc_std, wc_n = compute_within_cell_spearman(
            seed0["test_preds"], seed0["test_trues"], test_df["ModelID"].values
        )

        pred_drug_series = compute_drug_only_predictions(train_df, test_df)
        valid_mask = pred_drug_series.notna()
        wc_drug_mean, wc_drug_std, wc_drug_n = compute_within_cell_spearman(
            pred_drug_series[valid_mask].values,
            test_df.loc[valid_mask, "LN_IC50"].values,
            test_df.loc[valid_mask, "ModelID"].values,
        )

        results[split] = {
            "baselines": baselines,
            "model_global": {
                "mean": float(np.mean([r["test_spearman"] for r in seed_runs])),
                "std": float(np.std([r["test_spearman"] for r in seed_runs])),
                "values": [float(r["test_spearman"]) for r in seed_runs],
            },
            "model_rmse": {
                "mean": float(np.mean([r["test_rmse"] for r in seed_runs])),
                "std": float(np.std([r["test_rmse"] for r in seed_runs])),
                "values": [float(r["test_rmse"]) for r in seed_runs],
            },
            "model_within_cell": {
                "mean": wc_mean,
                "std": wc_std,
                "n_cells": wc_n,
            },
            "within_cell_drug_only": {
                "mean": wc_drug_mean,
                "std": wc_drug_std,
                "n_cells": wc_drug_n,
            },
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
        print(
            f"\nSaved OOD results to {OOD_RESULTS_PKL} and predictions to {OOD_PREDS_PKL}"
        )

    return results, preds_store


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
        preds_store = {}
        for split in SPLIT_NAMES:
            test_path = get_split_data_paths(split)["test"]
            test_df = pd.read_pickle(test_path).reset_index(drop=True)
            preds_store[split] = {
                "test_df": test_df,
                "preds": results[split]["test_preds"],
                "trues": results[split]["test_trues"],
                "spearman": results[split]["test_spearman"],
                "rmse": results[split]["test_rmse"],
            }
        with open(ood_preds_path, "wb") as f_pred:
            pickle.dump(preds_store, f_pred)

    return results, preds_store
