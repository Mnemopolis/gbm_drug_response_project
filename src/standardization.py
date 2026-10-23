"""
Leakage-free feature standardization, Principal Component Analysis (PCA),
imputation, and target Z-score normalization module.

Single Responsibility Principle (SRP):
Guarantees that all scaling statistics (mean, standard deviation, PCA components,
and median imputation values) are estimated strictly on the training split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from src.config import DEFAULT_SEED, N_PCA_COMPONENTS


def _fit_transform_pca_numpy(
    x_train_raw: np.ndarray,
    x_test_raw: np.ndarray,
    n_components: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pure NumPy Z-score standardization and SVD-based PCA fallback when scikit-learn is absent."""
    mean = x_train_raw.mean(axis=0, keepdims=True)
    std = x_train_raw.std(axis=0, keepdims=True)
    std_safe = np.where(std < 1e-8, 1.0, std)

    x_tr_scaled = (x_train_raw - mean) / std_safe
    x_te_scaled = (x_test_raw - mean) / std_safe

    pca_mean = x_tr_scaled.mean(axis=0, keepdims=True)
    x_tr_centered = x_tr_scaled - pca_mean
    x_te_centered = x_te_scaled - pca_mean

    _, _, vt = np.linalg.svd(x_tr_centered, full_matrices=False)
    components = vt[:n_components]
    return (
        (x_tr_centered @ components.T).astype(np.float32),
        (x_te_centered @ components.T).astype(np.float32),
    )


def fit_transform_cell_pca_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    expr_cols: list[str],
    mut_cols: list[str],
    mgmt_cols: list[str],
    n_pca: int = N_PCA_COMPONENTS,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit StandardScaler and PCA strictly on unique training cell lines to prevent
    data leakage, then transform both train and test cell line features.
    """
    cell_cols = expr_cols + mut_cols + mgmt_cols

    train_cells = (
        train_df[["ModelID"] + cell_cols]
        .drop_duplicates(subset=["ModelID"])
        .set_index("ModelID")
    )
    test_cells = (
        test_df[["ModelID"] + cell_cols]
        .drop_duplicates(subset=["ModelID"])
        .set_index("ModelID")
    )

    x_tr_raw = train_cells[expr_cols].astype(np.float32).fillna(0.0).values
    x_te_raw = test_cells[expr_cols].astype(np.float32).fillna(0.0).values

    max_pca = min(n_pca, x_tr_raw.shape[0] - 1, x_tr_raw.shape[1])
    if max_pca < 1:
        raise ValueError(
            f"Insufficient training cell lines for PCA: shape={train_cells.shape}"
        )

    if HAS_SKLEARN:
        scaler = StandardScaler()
        x_train_expr = scaler.fit_transform(x_tr_raw)
        x_test_expr = scaler.transform(x_te_raw)
        pca = PCA(n_components=max_pca, random_state=seed)
        train_pca = pca.fit_transform(x_train_expr)
        test_pca = pca.transform(x_test_expr)
    else:
        train_pca, test_pca = _fit_transform_pca_numpy(x_tr_raw, x_te_raw, max_pca)

    pca_col_names = [f"PCA_{i}" for i in range(max_pca)]
    train_pca_df = pd.DataFrame(
        train_pca, index=train_cells.index, columns=pca_col_names
    )
    test_pca_df = pd.DataFrame(test_pca, index=test_cells.index, columns=pca_col_names)

    train_mut_df = train_cells[mut_cols].astype(np.float32).fillna(0.0)
    test_mut_df = test_cells[mut_cols].astype(np.float32).fillna(0.0)

    if mgmt_cols:
        train_median = train_cells[mgmt_cols].median().fillna(0.0)
        train_mgmt_df = train_cells[mgmt_cols].fillna(train_median).astype(np.float32)
        test_mgmt_df = test_cells[mgmt_cols].fillna(train_median).astype(np.float32)
    else:
        train_mgmt_df = pd.DataFrame(index=train_cells.index)
        test_mgmt_df = pd.DataFrame(index=test_cells.index)

    train_cell_features = pd.concat([train_pca_df, train_mut_df, train_mgmt_df], axis=1)
    test_cell_features = pd.concat([test_pca_df, test_mut_df, test_mgmt_df], axis=1)

    train_final = train_df.drop(columns=cell_cols).merge(
        train_cell_features, left_on="ModelID", right_index=True, how="left"
    )
    test_final = test_df.drop(columns=cell_cols).merge(
        test_cell_features, left_on="ModelID", right_index=True, how="left"
    )

    return train_final, test_final


def standardize_target_labels(
    y_raw: np.ndarray,
    y_mean: float | None = None,
    y_std: float | None = None,
) -> tuple[np.ndarray, float, float]:
    """Z-score standardize target LN_IC50 values using training set mean and std."""
    y_arr = np.asarray(y_raw, dtype=np.float32)
    if y_mean is None or y_std is None:
        y_mean = float(y_arr.mean())
        raw_std = float(y_arr.std())
        y_std = raw_std if raw_std > 1e-8 else 1.0
    standardized = (y_arr - y_mean) / y_std
    return standardized.astype(np.float32), float(y_mean), float(y_std)


def unstandardize_target_labels(
    y_standardized: np.ndarray,
    y_mean: float,
    y_std: float,
) -> np.ndarray:
    """Invert Z-score standardization back to original LN_IC50 log-micromolar units."""
    return np.asarray(y_standardized, dtype=np.float32) * y_std + y_mean
