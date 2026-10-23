"""
Post-Hoc Error Analysis, Clinical Subtype Annotation, Scaffold Novelty Analysis,
and Publication-Ready Visualization Module (Notebooks 02, 03, and 04).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    from rdkit.Chem import rdFingerprintGenerator
except ImportError:
    rdFingerprintGenerator = None  # type: ignore

from src.baselines import evaluate_nearest_drug_scaffold_baseline
from src.config import (
    GBM_DRUG_KEYWORDS,
    KEY_GBM_DRUGS_FOR_INSPECTION,
    MODEL_FILE,
    MORGAN_FP_SIZE,
    MORGAN_RADIUS,
    RESULTS_DIR,
    SMILES_FILE,
    SPLIT_NAMES,
    get_split_data_paths,
)
from src.fingerprints import compute_max_tanimoto_similarity, compute_scaffold_morgan_fp


def get_top_k_prediction_errors(
    split_store: dict[str, Any], k: int = 10
) -> pd.DataFrame:
    """Return the top-K (drug, cell line) test pairs with the largest absolute prediction error."""
    df = split_store["test_df"].copy()
    df["pred"] = split_store["preds"]
    df["true"] = split_store["trues"]
    df["abs_err"] = (df["pred"] - df["true"]).abs()
    return df.nlargest(k, "abs_err")[
        ["DRUG_NAME", "ModelID", "true", "pred", "abs_err"]
    ].reset_index(drop=True)


def compute_per_drug_error_stats(
    split_store: dict[str, Any], min_n: int = 3
) -> pd.DataFrame:
    """Aggregate RMSE, mean signed error, and within-drug Spearman correlation per drug."""
    df = split_store["test_df"].copy()
    df["pred"] = split_store["preds"]
    df["true"] = split_store["trues"]
    df["err"] = df["pred"] - df["true"]

    rows: list[dict[str, Any]] = []
    for drug_name, group in df.groupby("DRUG_NAME"):
        if len(group) < min_n:
            continue
        true_std = float(group["true"].std()) if len(group) > 1 else 0.0
        if len(group) >= 3 and true_std >= 1e-6 and group["pred"].std() >= 1e-6:
            sp_val = float(spearmanr(group["pred"], group["true"]).correlation)
        else:
            sp_val = float("nan")

        rows.append(
            {
                "DRUG_NAME": drug_name,
                "n": len(group),
                "rmse": float(np.sqrt(np.mean(group["err"] ** 2))),
                "mean_err": float(group["err"].mean()),
                "spearman": sp_val if not np.isnan(sp_val) else np.nan,
                "true_std": true_std,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["DRUG_NAME", "n", "rmse", "mean_err", "spearman", "true_std"]
        )
    return (
        pd.DataFrame(rows).sort_values("rmse", ascending=False).reset_index(drop=True)
    )


def find_gbm_drugs_in_top_errors(
    per_drug_df: pd.DataFrame,
    top_n: int = 20,
    keywords: list[str] = GBM_DRUG_KEYWORDS,
) -> list[str]:
    """Check whether any clinically relevant GBM drugs appear among the top-N highest-RMSE drugs."""
    top_drugs = per_drug_df.head(top_n)["DRUG_NAME"].tolist()
    return [
        drug
        for drug in top_drugs
        if any(kw.lower() in str(drug).lower() for kw in keywords)
    ]


def compute_per_cell_error_stats(
    split_store: dict[str, Any], min_n: int = 20
) -> pd.DataFrame:
    """Aggregate RMSE, mean signed error, and within-cell Spearman correlation per cell line."""
    df = split_store["test_df"].copy()
    df["pred"] = split_store["preds"]
    df["true"] = split_store["trues"]
    df["err"] = df["pred"] - df["true"]

    rows: list[dict[str, Any]] = []
    for cell_id, group in df.groupby("ModelID"):
        if len(group) < min_n:
            continue
        sp_val = float(spearmanr(group["pred"], group["true"]).correlation)
        rows.append(
            {
                "ModelID": cell_id,
                "n": len(group),
                "rmse": float(np.sqrt(np.mean(group["err"] ** 2))),
                "mean_err": float(group["err"].mean()),
                "spearman": sp_val if not np.isnan(sp_val) else np.nan,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["ModelID", "n", "rmse", "mean_err", "spearman"])
    return (
        pd.DataFrame(rows).sort_values("rmse", ascending=False).reset_index(drop=True)
    )


def annotate_cells_with_clinical_subtypes(
    cell_stats_df: pd.DataFrame,
    model_file: Path = MODEL_FILE,
) -> pd.DataFrame:
    """Merge per-cell error statistics with DepMap OncotreeSubtype (Glioblastoma vs Gliosarcoma)."""
    if not model_file.exists() or len(cell_stats_df) == 0:
        return cell_stats_df

    model_full = pd.read_csv(model_file)
    available_cols = [
        col
        for col in ["ModelID", "CellLineName", "OncotreeCode", "OncotreeSubtype"]
        if col in model_full.columns
    ]
    return cell_stats_df.merge(model_full[available_cols], on="ModelID", how="left")


def analyze_scaffold_novelty_vs_error(
    scaffold_store: dict[str, Any],
    train_scaffold_pkl: Path | None = None,
    smiles_file: Path = SMILES_FILE,
) -> pd.DataFrame:
    """
    Compute the maximum Tanimoto similarity between each scaffold-split test drug
    and all training scaffolds, paired with the drug's test RMSE.
    """
    if rdFingerprintGenerator is None:
        raise ImportError("RDKit is required for analyze_scaffold_novelty_vs_error().")

    if train_scaffold_pkl is None:
        train_scaffold_pkl = get_split_data_paths("scaffold")["train"]

    drug_raw = pd.read_csv(smiles_file)
    drug_lookup = (
        drug_raw[["drug_name", "canonical_smiles"]]
        .dropna()
        .drop_duplicates(subset=["drug_name"])
        .rename(columns={"drug_name": "DRUG_NAME", "canonical_smiles": "SMILES"})
    )

    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS, fpSize=MORGAN_FP_SIZE
    )

    trainval_df = pd.read_pickle(train_scaffold_pkl)
    tr_drugs = (
        trainval_df[["DRUG_NAME"]]
        .drop_duplicates("DRUG_NAME")
        .merge(drug_lookup, on="DRUG_NAME", how="left")
    )
    tr_drugs["fp"] = tr_drugs["SMILES"].map(
        lambda s: compute_scaffold_morgan_fp(s, morgan_gen)
    )
    tr_drugs = tr_drugs[tr_drugs["fp"].notna()].copy()

    test_df = scaffold_store["test_df"].copy()
    te_drugs = (
        test_df[["DRUG_NAME"]]
        .drop_duplicates("DRUG_NAME")
        .merge(drug_lookup, on="DRUG_NAME", how="left")
    )
    te_drugs["fp"] = te_drugs["SMILES"].map(
        lambda s: compute_scaffold_morgan_fp(s, morgan_gen)
    )
    te_drugs = te_drugs[te_drugs["fp"].notna()].copy()

    ref_fps = tr_drugs["fp"].tolist()
    te_drugs["max_sim"] = te_drugs["fp"].map(
        lambda fp: compute_max_tanimoto_similarity(fp, ref_fps)
    )

    test_df["pred"] = scaffold_store["preds"]
    test_df["true"] = scaffold_store["trues"]
    test_df["sq_err"] = (test_df["pred"] - test_df["true"]) ** 2
    drug_rmse = (
        test_df.groupby("DRUG_NAME", as_index=False)["sq_err"]
        .mean()
        .assign(rmse=lambda d: np.sqrt(d["sq_err"]))[["DRUG_NAME", "rmse"]]
    )

    return (
        te_drugs[["DRUG_NAME", "SMILES", "max_sim"]]
        .merge(drug_rmse, on="DRUG_NAME", how="inner")
        .dropna(subset=["max_sim", "rmse"])
        .reset_index(drop=True)
    )


def inspect_key_gbm_drugs(
    preds_store: dict[str, Any],
    drug_keywords: list[str] = KEY_GBM_DRUGS_FOR_INSPECTION,
) -> pd.DataFrame:
    """Summarize true vs. predicted LN_IC50 and RMSE for key clinical GBM drugs across splits."""
    records = []
    for split in SPLIT_NAMES:
        if split not in preds_store:
            continue
        p = preds_store[split]
        df = p["test_df"].copy()
        df["pred"] = p["preds"]
        df["true"] = p["trues"]
        df["err"] = df["pred"] - df["true"]

        for kw in drug_keywords:
            sub = df[df["DRUG_NAME"].str.contains(kw, case=False, na=False)]
            if len(sub) == 0:
                continue
            records.append(
                {
                    "split": split,
                    "drug_keyword": kw,
                    "n": len(sub),
                    "true_mean": float(sub["true"].mean()),
                    "true_std": float(sub["true"].std()) if len(sub) > 1 else 0.0,
                    "pred_mean": float(sub["pred"].mean()),
                    "pred_std": float(sub["pred"].std()) if len(sub) > 1 else 0.0,
                    "mean_err": float(sub["err"].mean()),
                    "rmse": float(np.sqrt(np.mean(sub["err"] ** 2))),
                }
            )
    return pd.DataFrame(records)


# ==============================================================================
# Plotting Functions
# ==============================================================================


def plot_random_split_baseline_decomposition(
    random_results: dict[str, Any],
    output_path: Path = RESULTS_DIR / "diag_spearman_decomposition.png",
) -> Path:
    """Plot horizontal bar chart comparing statistical baselines vs. MLP on the random split."""
    baselines = random_results["baselines"]
    labels = [
        "Cell-only",
        "Drug-only",
        "Additive (no interaction)",
        "Model - global",
        "Model - within-cell",
    ]
    values = [
        baselines["cell_only"],
        baselines["drug_only"],
        baselines["additive"],
        random_results["model_global"]["mean"],
        random_results["model_within_cell"]["mean"],
    ]
    colors = ["#9e9e9e", "#9e9e9e", "#757575", "#1976d2", "#d32f2f"]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(labels, values, color=colors)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Spearman correlation (test set)")
    ax.set_title("Baseline decomposition on random split")
    ax.axvline(0, color="k", linewidth=0.8)

    for bar, val in zip(bars, values):
        if not np.isnan(val):
            ax.text(
                val + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center",
                fontsize=10,
            )

    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_three_split_ood_comparison(
    results: dict[str, Any],
    output_path: Path = RESULTS_DIR / "OOD_three_split_comparison.png",
) -> Path:
    """Plot grouped bar chart comparing baselines and model Spearman across all three splits."""
    labels = [
        "Cell-only",
        "Drug-only",
        "Additive",
        "Model (global)",
        "Model (within-cell)",
    ]
    data: dict[str, list[float]] = {label: [] for label in labels}
    errs: dict[str, list[float]] = {label: [] for label in labels}

    for split in SPLIT_NAMES:
        r = results[split]
        data["Cell-only"].append(np.nan_to_num(r["baselines"]["cell_only"], nan=0.0))
        errs["Cell-only"].append(0.0)
        data["Drug-only"].append(np.nan_to_num(r["baselines"]["drug_only"], nan=0.0))
        errs["Drug-only"].append(0.0)
        data["Additive"].append(np.nan_to_num(r["baselines"]["additive"], nan=0.0))
        errs["Additive"].append(0.0)
        data["Model (global)"].append(r["model_global"]["mean"])
        errs["Model (global)"].append(r["model_global"]["std"])
        data["Model (within-cell)"].append(r["model_within_cell"]["mean"])
        errs["Model (within-cell)"].append(r["model_within_cell"]["std"])

    x = np.arange(len(SPLIT_NAMES))
    width = 0.15
    colors = ["#bdbdbd", "#9e9e9e", "#757575", "#1976d2", "#d32f2f"]

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, label in enumerate(labels):
        ax.bar(
            x + (i - 2) * width,
            data[label],
            width,
            yerr=errs[label],
            capsize=3,
            label=label,
            color=colors[i],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            "Random\n(in-distribution)",
            "Leave-Cell-Out\n(cell-line OOD)",
            "Scaffold\n(chemical OOD)",
        ]
    )
    ax.set_ylabel("Spearman (test set)")
    ax.set_title("Model vs Baselines across Splits - GBM Drug Response")
    ax.set_ylim(0, 1.0)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_scatter_predictions_three_splits(
    preds_store: dict[str, Any],
    output_path: Path = RESULTS_DIR / "OOD_scatter_three_splits.png",
) -> Path:
    """Plot predicted vs. true LN_IC50 scatter plots across all three splits."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    titles = [
        "Random (in-distribution)",
        "Leave-Cell-Out (cell OOD)",
        "Scaffold (chemical OOD)",
    ]

    for ax, split, title in zip(axes, SPLIT_NAMES, titles):
        p = preds_store[split]
        preds = np.asarray(p["preds"])
        trues = np.asarray(p["trues"])

        ax.scatter(trues, preds, alpha=0.35, s=12, color="#1976d2", edgecolor="none")
        lims = [
            float(min(trues.min(), preds.min())) - 0.3,
            float(max(trues.max(), preds.max())) + 0.3,
        ]
        ax.plot(lims, lims, "k--", linewidth=1, label="y = x")
        ax.axhline(
            trues.mean(),
            color="#d32f2f",
            linestyle=":",
            linewidth=1,
            label=f"mean(y) = {trues.mean():.2f}",
        )
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("True LN_IC50")
        ax.set_ylabel("Predicted LN_IC50")
        ax.set_title(f"{title}\nSpearman={p['spearman']:.3f}, RMSE={p['rmse']:.3f}")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_scaffold_novelty_vs_error(
    novelty_df: pd.DataFrame,
    output_path: Path = RESULTS_DIR / "OOD_scaffold_novelty_vs_error.png",
) -> Path:
    """Plot per-drug test RMSE against maximum Tanimoto similarity to training scaffolds."""
    corr = float(novelty_df[["max_sim", "rmse"]].corr(method="spearman").iloc[0, 1])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        novelty_df["max_sim"],
        novelty_df["rmse"],
        s=30,
        alpha=0.6,
        color="#d32f2f",
        edgecolor="k",
        linewidth=0.3,
    )
    ax.set_xlabel("Max Tanimoto similarity to any training scaffold")
    ax.set_ylabel("Per-drug RMSE (scaffold split)")
    ax.set_title(
        f"Scaffold Novelty vs Prediction Error\nSpearman(Similarity, RMSE) = {corr:.3f}"
    )
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_per_drug_spearman_scaffold_boxplot(
    preds_store: dict[str, Any],
    output_path: Path = RESULTS_DIR / "OOD_per_drug_spearman_scaffold.png",
) -> Path:
    """Plot boxplot of within-drug Spearman correlations across cells on the scaffold split."""
    pstats = compute_per_drug_error_stats(preds_store["scaffold"], min_n=10)
    valid_sp = pstats["spearman"].dropna().values

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.boxplot(
        [valid_sp],
        tick_labels=["Scaffold"],
        patch_artist=True,
        boxprops={"facecolor": "#d32f2f", "alpha": 0.6},
    )
    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    ax.set_ylabel("Per-drug Spearman\n(within drug, across cells)")
    ax.set_title(f"Per-drug Spearman - Scaffold split (n={len(valid_sp)} drugs)")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_per_drug_rmse_boxplot_across_splits(
    preds_store: dict[str, Any],
    output_path: Path = RESULTS_DIR / "OOD_per_drug_rmse_box.png",
) -> Path:
    """Plot per-drug RMSE distribution boxplots across Random, Leave-Cell, and Scaffold splits."""
    rmse_data = []
    labels = []

    for split in SPLIT_NAMES:
        pstats = compute_per_drug_error_stats(preds_store[split], min_n=1)
        if len(pstats) == 0:
            continue
        rmse_data.append(pstats["rmse"].values)
        labels.append(f"{split}\n(n={len(pstats)} drugs)")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bp = ax.boxplot(
        rmse_data,
        tick_labels=labels,
        patch_artist=True,
        showfliers=False,
    )
    colors = ["#1976d2", "#f57c00", "#d32f2f"]
    for patch, color in zip(bp["boxes"], colors[: len(rmse_data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("Per-drug RMSE (lower = better)")
    ax.set_title("Per-drug RMSE distribution across splits")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_prediction_bias_boxplot_across_splits(
    preds_store: dict[str, Any],
    output_path: Path = RESULTS_DIR / "OOD_prediction_bias_box.png",
) -> Path:
    """Plot signed prediction error (pred - true) boxplots across all three splits."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#1976d2", "#f57c00", "#d32f2f"]

    for idx, (split, color) in enumerate(zip(SPLIT_NAMES, colors)):
        p = preds_store[split]
        err = np.asarray(p["preds"]) - np.asarray(p["trues"])
        ax.boxplot(
            [err],
            positions=[idx],
            widths=0.5,
            patch_artist=True,
            showfliers=False,
            boxprops={"facecolor": color, "alpha": 0.6},
        )

    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(SPLIT_NAMES)))
    ax.set_xticklabels(SPLIT_NAMES)
    ax.set_ylabel("Prediction error (pred - true, log units)")
    ax.set_title("Systematic bias across splits")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_global_vs_within_drug_spearman(
    preds_store: dict[str, Any],
    output_path: Path = RESULTS_DIR / "OOD_global_vs_within_drug_spearman.png",
) -> Path:
    """Plot side-by-side comparison of global Spearman vs. median within-drug Spearman."""
    global_sp = []
    median_wc = []

    for split in SPLIT_NAMES:
        min_n = 10 if split == "scaffold" else 3
        pstats = compute_per_drug_error_stats(preds_store[split], min_n=min_n)
        global_sp.append(float(preds_store[split]["spearman"]))
        median_wc.append(float(pstats["spearman"].median()))

    x = np.arange(len(SPLIT_NAMES))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(
        x - width / 2,
        global_sp,
        width,
        label="Global Spearman",
        color="#1976d2",
        alpha=0.7,
    )
    ax.bar(
        x + width / 2,
        median_wc,
        width,
        label="Median per-drug Spearman",
        color="#d32f2f",
        alpha=0.7,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(SPLIT_NAMES)
    ax.set_ylabel("Spearman")
    ax.set_title("Global vs Within-drug Spearman - decoupling across splits")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def run_results_analysis_workflow(
    results: dict[str, Any], preds_store: dict[str, Any]
) -> list[Path]:
    """Execute all post-hoc error analyses and generate all diagnostic figures in `Results/`."""
    print("\n" + "=" * 70)
    print("WORKFLOW 4: Running Post-Hoc Error Analysis & Generating Figures")
    print("=" * 70)

    scaf_paths = get_split_data_paths("scaffold")
    if scaf_paths["train"].exists() and scaf_paths["test"].exists():
        tr_scaf = pd.read_pickle(scaf_paths["train"])
        te_scaf = pd.read_pickle(scaf_paths["test"])
        sp_nn = evaluate_nearest_drug_scaffold_baseline(tr_scaf, te_scaf)
        print(
            f"\n[Scaffold Split] Nearest-Drug Morgan FP Baseline Spearman = {sp_nn:.4f}"
        )

    for split in SPLIT_NAMES:
        print(f"\n--- [{split}] Top-10 Largest Absolute Prediction Errors ---")
        top10 = get_top_k_prediction_errors(preds_store[split], k=10)
        print(top10.to_string(index=False))

    for split in SPLIT_NAMES:
        pstats = compute_per_drug_error_stats(preds_store[split], min_n=3)
        gbm_hits = find_gbm_drugs_in_top_errors(
            compute_per_drug_error_stats(preds_store[split], min_n=10), top_n=20
        )
        print(
            f"\n[{split}] Valid drugs (n>=3): {len(pstats)} | "
            f"Median RMSE={pstats['rmse'].median():.3f} | "
            f"Median Per-Drug Spearman={pstats['spearman'].median():.3f} | "
            f"Top-20 High-Error GBM Drugs={gbm_hits if gbm_hits else 'None'}"
        )

    for split in SPLIT_NAMES:
        cstats = compute_per_cell_error_stats(preds_store[split], min_n=20)
        annotated = annotate_cells_with_clinical_subtypes(cstats, MODEL_FILE)
        print(f"\n[{split}] Per-Cell Summary (n={len(annotated)} cells):")
        print(annotated.head(5).to_string(index=False))

    gbm_drug_summary = inspect_key_gbm_drugs(preds_store)
    if len(gbm_drug_summary) > 0:
        print("\n--- Key Clinical GBM Drugs Performance ---")
        print(gbm_drug_summary.to_string(index=False))

    saved_figs: list[Path] = []
    if "random" in results:
        saved_figs.append(plot_random_split_baseline_decomposition(results["random"]))
    saved_figs.append(plot_three_split_ood_comparison(results))
    saved_figs.append(plot_scatter_predictions_three_splits(preds_store))
    saved_figs.append(plot_per_drug_spearman_scaffold_boxplot(preds_store))
    saved_figs.append(plot_per_drug_rmse_boxplot_across_splits(preds_store))
    saved_figs.append(plot_prediction_bias_boxplot_across_splits(preds_store))
    saved_figs.append(plot_global_vs_within_drug_spearman(preds_store))

    if SMILES_FILE.exists() and scaf_paths["train"].exists():
        novelty_df = analyze_scaffold_novelty_vs_error(preds_store["scaffold"])
        saved_figs.append(plot_scaffold_novelty_vs_error(novelty_df))

    print("\nGenerated Figures in Results/:")
    for fig_path in saved_figs:
        print(f"  - {fig_path.name}")

    return saved_figs
