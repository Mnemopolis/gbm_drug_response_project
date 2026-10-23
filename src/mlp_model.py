"""
Two-Tower Multi-Layer Perceptron (`BaselineMLP`) neural network architecture
for multimodal Glioblastoma (GBM) drug response prediction.

Single Responsibility Principle (SRP):
Defines only the neural network architecture and forward pass.
"""

try:
    import torch
    from torch import nn
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore

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
