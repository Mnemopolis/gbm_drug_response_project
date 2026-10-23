"""
PyTorch Dataset, PyTorch Geometric Molecular Graph Dataset, and GPU DataLoader module.

Single Responsibility Principle (SRP):
Handles tensor wrapping, molecular graph construction (`smiles_to_pyg_graph`),
and GPU-accelerated DataLoader instantiation (`pin_memory=True`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError:
    torch = None  # type: ignore
    DataLoader = object  # type: ignore
    Dataset = object  # type: ignore

try:
    from rdkit import Chem
except ImportError:
    Chem = None  # type: ignore

try:
    from torch_geometric.data import Data as PyGData
except ImportError:
    PyGData = None  # type: ignore

from src.config import ATOM_TO_IDX, ATOM_TYPES, BATCH_SIZE
from src.standardization import standardize_target_labels


def get_cell_feature_columns(df: pd.DataFrame) -> list[str]:
    """Identify all engineered cell-line multi-omics feature columns (PCA_, MUT_, MGMT_)."""
    return [col for col in df.columns if col.startswith(("PCA_", "MUT_", "MGMT_"))]


class DrugResponseDataset(Dataset):
    """
    PyTorch Dataset for multimodal drug response prediction using 2048-bit Morgan
    fingerprints and PCA/mutation/MGMT cell line features with Z-score target scaling.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        y_mean: float | None = None,
        y_std: float | None = None,
    ) -> None:
        if torch is None:
            raise ImportError("PyTorch is required to instantiate DrugResponseDataset.")
        super().__init__()
        self.cell_cols: list[str] = get_cell_feature_columns(df)
        self.fingerprints = torch.tensor(
            np.stack(df["Morgan_FP"].values), dtype=torch.float32
        )
        self.cell_features = torch.tensor(
            df[self.cell_cols].values.astype(np.float32), dtype=torch.float32
        )

        y_scaled, self.y_mean, self.y_std = standardize_target_labels(
            df["LN_IC50"].values, y_mean=y_mean, y_std=y_std
        )
        self.labels = torch.tensor(y_scaled, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.fingerprints[idx], self.cell_features[idx], self.labels[idx]


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = BATCH_SIZE,
    pin_memory: bool | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, DrugResponseDataset]:
    """
    Create train, validation, and test PyTorch DataLoaders with shared target normalization.
    Automatically enables `pin_memory=True` when a CUDA GPU is available.
    """
    if torch is None:
        raise ImportError("PyTorch is required for create_dataloaders().")
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_ds = DrugResponseDataset(train_df)
    val_ds = DrugResponseDataset(val_df, y_mean=train_ds.y_mean, y_std=train_ds.y_std)
    test_ds = DrugResponseDataset(test_df, y_mean=train_ds.y_mean, y_std=train_ds.y_std)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader, train_ds


def smiles_to_pyg_graph(smiles: str):
    """
    Convert a SMILES string into a PyTorch Geometric `Data` molecular graph with
    16-dimensional atom node features and bidirectional chemical bond edges.
    """
    if PyGData is None or torch is None or Chem is None:
        raise ImportError(
            "torch, torch_geometric, and rdkit are required for smiles_to_pyg_graph()."
        )

    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None

    node_features: list[list[float]] = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        atom_idx = ATOM_TO_IDX.get(symbol, len(ATOM_TYPES))
        one_hot = [0.0] * (len(ATOM_TYPES) + 1)
        one_hot[atom_idx] = 1.0
        atom_props = [
            atom.GetDegree() / 4.0,
            float(atom.GetIsAromatic()),
            float(atom.GetFormalCharge()),
        ]
        node_features.append(one_hot + atom_props)

    x_tensor = torch.tensor(node_features, dtype=torch.float32)
    if mol.GetNumBonds() == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        sources, targets = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            sources.extend([i, j])
            targets.extend([j, i])
        edge_index = torch.tensor([sources, targets], dtype=torch.long)

    return PyGData(x=x_tensor, edge_index=edge_index)


class DrugResponseGraphDataset(Dataset):
    """PyTorch Geometric dataset pairing molecular graphs with cell-line omics features."""

    def __init__(
        self,
        df: pd.DataFrame,
        y_mean: float | None = None,
        y_std: float | None = None,
    ) -> None:
        if torch is None:
            raise ImportError("PyTorch is required for DrugResponseGraphDataset.")
        super().__init__()
        if "SMILES" not in df.columns:
            raise ValueError(
                "DataFrame must contain a 'SMILES' column for GNN dataset."
            )

        self.cell_cols: list[str] = get_cell_feature_columns(df)
        self.cell_features = torch.tensor(
            df[self.cell_cols].values.astype(np.float32), dtype=torch.float32
        )
        self.smiles_list: list[str] = df["SMILES"].tolist()
        self.graphs = [smiles_to_pyg_graph(smi) for smi in self.smiles_list]

        y_scaled, self.y_mean, self.y_std = standardize_target_labels(
            df["LN_IC50"].values, y_mean=y_mean, y_std=y_std
        )
        self.labels = torch.tensor(y_scaled, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.graphs[idx], self.cell_features[idx], self.labels[idx]
