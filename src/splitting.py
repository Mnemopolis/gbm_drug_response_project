"""
In-distribution and Out-of-Distribution (OOD) dataset splitting module.

Single Responsibility Principle (SRP):
Handles partitioning of drug-response dataframes into train, validation, and test
subsets under Random, Leave-Cell-Out, and Bemis-Murcko Scaffold regimes.
"""

import numpy as np
import pandas as pd

try:
    from sklearn.model_selection import train_test_split

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

from src.config import DEFAULT_SEED, TEST_SIZE, VAL_SIZE


def partition_dataframe_by_split(
    df: pd.DataFrame,
    split_type: str,
    test_size: float = TEST_SIZE,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Partition the merged dataframe into train and test subsets according to:
    - 'random': In-distribution random (drug, cell) holdout
    - 'leave_cell': Biological OOD holdout of unseen cell lines (ModelID)
    - 'scaffold': Chemical OOD holdout of unseen Bemis-Murcko scaffolds
    """
    if len(df) == 0:
        raise ValueError("Input dataframe is empty; cannot perform train/test split.")

    rng = np.random.default_rng(seed)

    if split_type == "random":
        if HAS_SKLEARN:
            train_df, test_df = train_test_split(
                df, test_size=test_size, random_state=seed
            )
            return train_df.copy(), test_df.copy()
        indices = np.arange(len(df))
        rng.shuffle(indices)
        n_test = max(1, round(len(df) * test_size))
        return df.iloc[indices[n_test:]].copy(), df.iloc[indices[:n_test]].copy()

    if split_type == "leave_cell":
        unique_cells = df["ModelID"].unique()
        n_test_cells = max(1, int(len(unique_cells) * test_size))
        test_cells = rng.choice(unique_cells, size=n_test_cells, replace=False)
        train_df = df[~df["ModelID"].isin(test_cells)].copy()
        test_df = df[df["ModelID"].isin(test_cells)].copy()
        return train_df, test_df

    if split_type == "scaffold":
        unique_scaffolds = df["Scaffold"].dropna().unique()
        n_test_scaffolds = max(1, int(len(unique_scaffolds) * test_size))
        test_scaffolds = rng.choice(
            unique_scaffolds, size=n_test_scaffolds, replace=False
        )
        train_df = df[~df["Scaffold"].isin(test_scaffolds)].copy()
        test_df = df[df["Scaffold"].isin(test_scaffolds)].copy()
        return train_df, test_df

    raise ValueError(f"Unsupported split_type: '{split_type}'.")


def partition_train_validation(
    trainval_df: pd.DataFrame,
    val_size: float = VAL_SIZE,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the training/validation dataframe into train and validation sets,
    stratifying by cell line (`ModelID`) when possible so validation covers all training cells.
    """
    if HAS_SKLEARN:
        try:
            train_df, val_df = train_test_split(
                trainval_df,
                test_size=val_size,
                random_state=seed,
                stratify=trainval_df["ModelID"],
            )
            return train_df.copy(), val_df.copy()
        except ValueError:
            train_df, val_df = train_test_split(
                trainval_df,
                test_size=val_size,
                random_state=seed,
            )
            return train_df.copy(), val_df.copy()

    rng = np.random.default_rng(seed)
    indices = np.arange(len(trainval_df))
    rng.shuffle(indices)
    n_val = max(1, round(len(trainval_df) * val_size))
    return trainval_df.iloc[indices[n_val:]].copy(), trainval_df.iloc[
        indices[:n_val]
    ].copy()
