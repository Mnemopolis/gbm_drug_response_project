"""
Neural network architectures for multimodal GBM drug response prediction.

Includes:
- `BaselineMLP`: Two-tower MLP encoding Morgan fingerprints and cell-line omics features.
- `GNNDrugMLP`: Graph Isomorphism Network (GINConv) drug encoder paired with an MLP cell encoder.
"""

try:
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore

try:
    from torch_geometric.nn import GINConv, global_mean_pool
except ImportError:
    GINConv = None  # type: ignore
    global_mean_pool = None  # type: ignore

from src.config import DROPOUT_RATE, HIDDEN_DIM, MORGAN_FP_SIZE

_BaseModule = nn.Module if nn is not None else object


class BaselineMLP(_BaseModule):
    """
    Two-tower Multi-Layer Perceptron (MLP) for drug sensitivity prediction.
    - Drug Tower: Maps 2048-bit Morgan fingerprint -> 512 -> hidden_dim
    - Cell Tower: Maps PCA/Mutation/MGMT features -> 128 -> hidden_dim
    - Fusion Predictor: Concatenates embeddings -> 128 -> scalar LN_IC50 prediction
    """

    def __init__(
        self,
        fp_dim: int = MORGAN_FP_SIZE,
        cell_dim: int = 44,
        hidden_dim: int = HIDDEN_DIM,
        dropout: float = DROPOUT_RATE,
    ) -> None:
        if nn is None:
            raise ImportError("PyTorch is required to instantiate BaselineMLP.")
        super().__init__()
        self.drug_net = nn.Sequential(
            nn.Linear(fp_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, hidden_dim),
        )
        self.cell_net = nn.Sequential(
            nn.Linear(cell_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, hidden_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, fp, cell):
        """Forward pass returning a 1D tensor of predicted standardized LN_IC50 values."""
        drug_emb = self.drug_net(fp)
        cell_emb = self.cell_net(cell)
        combined = torch.cat([drug_emb, cell_emb], dim=1)
        return self.predictor(combined).squeeze(-1)


class GNNDrugMLP(_BaseModule):
    """
    Two-tower Graph Neural Network + MLP architecture using GINConv layers
    to encode molecular graphs alongside cell-line omics features.
    """

    def __init__(
        self,
        node_dim: int = 16,
        cell_dim: int = 44,
        hidden_dim: int = HIDDEN_DIM,
        dropout: float = DROPOUT_RATE,
    ) -> None:
        if nn is None or GINConv is None or global_mean_pool is None:
            raise ImportError(
                "torch and torch_geometric are required to instantiate GNNDrugMLP."
            )
        super().__init__()
        self.conv1 = GINConv(
            nn.Sequential(
                nn.Linear(node_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
            )
        )
        self.conv2 = GINConv(
            nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, hidden_dim),
            )
        )
        self.cell_net = nn.Sequential(
            nn.Linear(cell_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, hidden_dim),
        )
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, graph_batch, cell):
        """Forward pass over a PyG Batch of molecular graphs and cell-line feature tensor."""
        x, edge_index, batch = graph_batch.x, graph_batch.edge_index, graph_batch.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        drug_emb = global_mean_pool(x, batch)
        cell_emb = self.cell_net(cell)
        combined = torch.cat([drug_emb, cell_emb], dim=1)
        return self.predictor(combined).squeeze(-1)
