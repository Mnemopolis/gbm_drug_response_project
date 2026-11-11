"""
Post-Hoc Error Analysis, Clinical Subtype Annotation, Scaffold Novelty Analysis,
and Publication-Ready Visualization Module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from rdkit.Chem import rdFingerprintGenerator
except ImportError:
    rdFingerprintGenerator = None  # type: ignore

from src.baselines import (
    compute_grouped_regression_stats,
    evaluate_nearest_drug_scaffold_baseline,
)
from src.config import (
    GBM_DRUG_KEYWORDS,
    KEY_GBM_DRUGS_FOR_INSPECTION,
    MODEL_FILE,
    MORGAN_FP_SIZE,
    MORGAN_RADIUS,
    RESULTS_DIR,
    SMILES_FILE,
    SPLIT_COLORS,
    SPLIT_DISPLAY_TITLES,
    SPLIT_NAMES,
    get_split_data_paths,
)
from src.deduplication import deduplicate_drug_smiles_table
from src.fingerprints import compute_max_tanimoto_similarity, compute_scaffold_morgan_fp


def _save_figure(fig: Any, output_path: Path, dpi: int = 150) -> Path:
    """Apply tight layout, save figure to `output_path`, and close the Matplotlib figure."""
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def _with_pred_columns(split_store: dict[str, Any]) -> pd.DataFrame:
    """Attach `pred`, `true`, and signed `err` columns to a copy of `split_store['test_df']`."""
    df = split_store["test_df"].copy()
    df["pred"] = np.asarray(split_store["preds"], dtype=np.float64)
    df["true"] = np.asarray(split_store["trues"], dtype=np.float64)
    df["err"] = df["pred"] - df["true"]
    return df


def get_top_k_prediction_errors(
    split_store: dict[str, Any], k: int = 10
) -> pd.DataFrame:
    """Return the top-K (drug, cell line) test pairs with the largest absolute prediction error."""
    df = _with_pred_columns(split_store)
    df["abs_err"] = df["err"].abs()
    return df.nlargest(k, "abs_err")[
        ["DRUG_NAME", "ModelID", "true", "pred", "abs_err"]
    ].reset_index(drop=True)


def compute_per_drug_error_stats(
    split_store: dict[str, Any], min_n: int = 3
) -> pd.DataFrame:
    """Aggregate RMSE, mean signed error, and within-drug Spearman correlation per drug."""
    return compute_grouped_regression_stats(
        _with_pred_columns(split_store),
        group_col="DRUG_NAME",
        min_n=min_n,
        include_true_std=True,
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
    return compute_grouped_regression_stats(
        _with_pred_columns(split_store),
        group_col="ModelID",
        min_n=min_n,
        include_true_std=False,
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


def _resolve_drug_smiles_df(
    df: pd.DataFrame, smiles_file: Path = SMILES_FILE
) -> pd.DataFrame:
    """Extract unique (DRUG_NAME, SMILES) directly from `df` if cached, else from `smiles_file`."""
    if "SMILES" in df.columns:
        return df[["DRUG_NAME", "SMILES"]].dropna().drop_duplicates("DRUG_NAME").copy()
    drug_lookup = deduplicate_drug_smiles_table(pd.read_csv(smiles_file))[
        ["DRUG_NAME", "SMILES"]
    ]
    return (
        df[["DRUG_NAME"]]
        .drop_duplicates("DRUG_NAME")
        .merge(drug_lookup, on="DRUG_NAME", how="left")
    )


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

    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS, fpSize=MORGAN_FP_SIZE
    )
    trainval_df = pd.read_pickle(train_scaffold_pkl)
    tr_drugs = _resolve_drug_smiles_df(trainval_df, smiles_file)
    tr_drugs["fp"] = tr_drugs["SMILES"].map(
        lambda s: compute_scaffold_morgan_fp(s, morgan_gen)
    )
    ref_fps = tr_drugs["fp"].dropna().tolist()

    te_drugs = _resolve_drug_smiles_df(scaffold_store["test_df"], smiles_file)
    te_drugs["fp"] = te_drugs["SMILES"].map(
        lambda s: compute_scaffold_morgan_fp(s, morgan_gen)
    )
    te_drugs = te_drugs[te_drugs["fp"].notna()].copy()
    te_drugs["max_sim"] = te_drugs["fp"].map(
        lambda fp: compute_max_tanimoto_similarity(fp, ref_fps)
    )

    drug_rmse = compute_per_drug_error_stats(scaffold_store, min_n=1)[
        ["DRUG_NAME", "rmse"]
    ]
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
        df = _with_pred_columns(preds_store[split])
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
    return _save_figure(fig, output_path)


def plot_three_split_ood_comparison(
    results: dict[str, Any],
    output_path: Path = RESULTS_DIR / "OOD_three_split_comparison.png",
) -> Path:
    """Plot grouped bar chart comparing baselines and model Spearman across all three splits."""
    series_specs = [
        ("Cell-only", "#bdbdbd", lambda r: (np.nan_to_num(r["baselines"]["cell_only"], nan=0.0), 0.0)),
        ("Drug-only", "#9e9e9e", lambda r: (np.nan_to_num(r["baselines"]["drug_only"], nan=0.0), 0.0)),
        ("Additive", "#757575", lambda r: (np.nan_to_num(r["baselines"]["additive"], nan=0.0), 0.0)),
        ("Model (global)", "#1976d2", lambda r: (r["model_global"]["mean"], r["model_global"]["std"])),
        ("Model (within-cell)", "#d32f2f", lambda r: (r["model_within_cell"]["mean"], r["model_within_cell"]["std"])),
    ]
    x = np.arange(len(SPLIT_NAMES))
    width = 0.15

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (label, color, extractor) in enumerate(series_specs):
        means_stds = [extractor(results[s]) for s in SPLIT_NAMES]
        ax.bar(
            x + (i - 2) * width,
            [m for m, _ in means_stds],
            width,
            yerr=[s for _, s in means_stds],
            capsize=3,
            label=label,
            color=color,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([SPLIT_DISPLAY_TITLES[s].replace(" (", "\n(") for s in SPLIT_NAMES])
    ax.set_ylabel("Spearman (test set)")
    ax.set_title("Model vs Baselines across Splits - GBM Drug Response")
    ax.set_ylim(0, 1.0)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    return _save_figure(fig, output_path)


def plot_scatter_predictions_three_splits(
    preds_store: dict[str, Any],
    output_path: Path = RESULTS_DIR / "OOD_scatter_three_splits.png",
) -> Path:
    """Plot predicted vs. true LN_IC50 scatter plots across all three splits."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, split in zip(axes, SPLIT_NAMES):
        p = preds_store[split]
        preds, trues = np.asarray(p["preds"]), np.asarray(p["trues"])
        lims = [
            float(min(trues.min(), preds.min())) - 0.3,
            float(max(trues.max(), preds.max())) + 0.3,
        ]
        ax.scatter(trues, preds, alpha=0.35, s=12, color="#1976d2", edgecolor="none")
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
        ax.set_title(
            f"{SPLIT_DISPLAY_TITLES[split]}\nSpearman={p['spearman']:.3f}, RMSE={p['rmse']:.3f}"
        )
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)
    return _save_figure(fig, output_path)


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
    return _save_figure(fig, output_path)


def plot_per_drug_spearman_scaffold_boxplot(
    preds_store: dict[str, Any],
    output_path: Path = RESULTS_DIR / "OOD_per_drug_spearman_scaffold.png",
) -> Path:
    """Plot boxplot of within-drug Spearman correlations across cells on the scaffold split."""
    valid_sp = (
        compute_per_drug_error_stats(preds_store["scaffold"], min_n=10)["spearman"]
        .dropna()
        .values
    )
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
    return _save_figure(fig, output_path)


def plot_per_drug_rmse_boxplot_across_splits(
    preds_store: dict[str, Any],
    output_path: Path = RESULTS_DIR / "OOD_per_drug_rmse_box.png",
) -> Path:
    """Plot per-drug RMSE distribution boxplots across Random, Leave-Cell, and Scaffold splits."""
    rmse_data, labels = [], []
    for split in SPLIT_NAMES:
        pstats = compute_per_drug_error_stats(preds_store[split], min_n=1)
        if len(pstats) > 0:
            rmse_data.append(pstats["rmse"].values)
            labels.append(f"{split}\n(n={len(pstats)} drugs)")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bp = ax.boxplot(rmse_data, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], SPLIT_COLORS[: len(rmse_data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel("Per-drug RMSE (lower = better)")
    ax.set_title("Per-drug RMSE distribution across splits")
    ax.grid(axis="y", alpha=0.3)
    return _save_figure(fig, output_path)


def plot_prediction_bias_boxplot_across_splits(
    preds_store: dict[str, Any],
    output_path: Path = RESULTS_DIR / "OOD_prediction_bias_box.png",
) -> Path:
    """Plot signed prediction error (pred - true) boxplots across all three splits."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for idx, (split, color) in enumerate(zip(SPLIT_NAMES, SPLIT_COLORS)):
        err = np.asarray(preds_store[split]["preds"]) - np.asarray(
            preds_store[split]["trues"]
        )
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
    return _save_figure(fig, output_path)


def plot_global_vs_within_drug_spearman(
    preds_store: dict[str, Any],
    output_path: Path = RESULTS_DIR / "OOD_global_vs_within_drug_spearman.png",
) -> Path:
    """Plot side-by-side comparison of global Spearman vs. median within-drug Spearman."""
    global_sp = [float(preds_store[s]["spearman"]) for s in SPLIT_NAMES]
    median_wc = [
        float(
            compute_per_drug_error_stats(
                preds_store[s], min_n=(10 if s == "scaffold" else 3)
            )["spearman"].median()
        )
        for s in SPLIT_NAMES
    ]
    x = np.arange(len(SPLIT_NAMES))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - width / 2, global_sp, width, label="Global Spearman", color="#1976d2", alpha=0.7)
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
    return _save_figure(fig, output_path)


def run_results_analysis_workflow(
    results: dict[str, Any], preds_store: dict[str, Any]
) -> list[Path]:
    """Execute all post-hoc error analyses and generate all diagnostic figures in `Results/`."""
    print("\n" + "=" * 70)
    print("WORKFLOW 4: Running Post-Hoc Error Analysis & Generating Figures")
    print("=" * 70)

    scaf_paths = get_split_data_paths("scaffold")
    if scaf_paths["train"].exists() and scaf_paths["test"].exists():
        sp_nn = evaluate_nearest_drug_scaffold_baseline(
            pd.read_pickle(scaf_paths["train"]), pd.read_pickle(scaf_paths["test"])
        )
        print(f"\n[Scaffold Split] Nearest-Drug Morgan FP Baseline Spearman = {sp_nn:.4f}")

    for split in SPLIT_NAMES:
        print(f"\n--- [{split}] Top-10 Largest Absolute Prediction Errors ---")
        print(get_top_k_prediction_errors(preds_store[split], k=10).to_string(index=False))

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

        cstats = annotate_cells_with_clinical_subtypes(
            compute_per_cell_error_stats(preds_store[split], min_n=20), MODEL_FILE
        )
        print(f"\n[{split}] Per-Cell Summary (n={len(cstats)} cells):")
        print(cstats.head(5).to_string(index=False))

    gbm_drug_summary = inspect_key_gbm_drugs(preds_store)
    if len(gbm_drug_summary) > 0:
        print("\n--- Key Clinical GBM Drugs Performance ---")
        print(gbm_drug_summary.to_string(index=False))

    saved_figs: list[Path] = []
    if "random" in results:
        saved_figs.append(plot_random_split_baseline_decomposition(results["random"]))
    saved_figs.extend(
        [
            plot_three_split_ood_comparison(results),
            plot_scatter_predictions_three_splits(preds_store),
            plot_per_drug_spearman_scaffold_boxplot(preds_store),
            plot_per_drug_rmse_boxplot_across_splits(preds_store),
            plot_prediction_bias_boxplot_across_splits(preds_store),
            plot_global_vs_within_drug_spearman(preds_store),
        ]
    )
    if scaf_paths["train"].exists():
        novelty_df = analyze_scaffold_novelty_vs_error(preds_store["scaffold"])
        saved_figs.append(plot_scaffold_novelty_vs_error(novelty_df))

    print("\nGenerated Figures in Results/:")
    for fig_path in saved_figs:
        print(f"  - {fig_path.name}")
    return saved_figs
