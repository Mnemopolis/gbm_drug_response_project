"""PyTorch Dataset and DataLoader construction for tabular drug-cell pairs.

Responsibilities:
- Construct PyTorch Datasets for Morgan fingerprints and cell-line multi-omics features.
- Provide DataLoader factory helpers with deterministic train/val/test batching.
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Dataset = object
    DataLoader = object

from src.config import BATCH_SIZE
from src.standardization import standardize_target_labels


class DrugResponseDataset(Dataset):
    """PyTorch Dataset yielding (morgan_fp, cell_features, standardized_ic50) tuples."""

    def __init__(
        self,
        df: pd.DataFrame,
        y_mean: Optional[float] = None,
        y_std: Optional[float] = None,
    ) -> None:
        if not HAS_TORCH:
            raise ImportError("PyTorch is required to instantiate DrugResponseDataset.")

        self.cell_cols: List[str] = [c for c in df.columns if c.startswith(("PCA_", "MUT_", "MGMT_"))]
        if not self.cell_cols:
            raise ValueError("No PCA_, MUT_, or MGMT_ columns found in DataFrame.")

        fp_array = np.stack(df["Morgan_FP"].values).astype(np.float32)
        cell_array = df[self.cell_cols].values.astype(np.float32)
        raw_y = df["LN_IC50"].values.astype(np.float32)

        std_y, self.y_mean, self.y_std = standardize_target_labels(raw_y, y_mean, y_std)

        self.fingerprints = torch.tensor(fp_array, dtype=torch.float32)
        self.cell_features = torch.tensor(cell_array, dtype=torch.float32)
        self.labels = torch.tensor(std_y, dtype=torch.float32)

    @property
    def cell_dim(self) -> int:
        """Return the dimensionality of the cell-line feature vector."""
        return len(self.cell_cols)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        return self.fingerprints[idx], self.cell_features[idx], self.labels[idx]


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = BATCH_SIZE,
) -> Tuple["DataLoader", "DataLoader", "DataLoader", DrugResponseDataset]:
    """Create train, validation, and test DataLoaders with shared training label statistics."""
    train_ds = DrugResponseDataset(train_df)
    val_ds = DrugResponseDataset(val_df, y_mean=train_ds.y_mean, y_std=train_ds.y_std)
    test_ds = DrugResponseDataset(test_df, y_mean=train_ds.y_mean, y_std=train_ds.y_std)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_ds
