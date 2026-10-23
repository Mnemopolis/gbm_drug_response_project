"""
Atomic neural network training, validation early stopping, and evaluation primitives.

Single Responsibility Principle (SRP):
Contains only single-epoch training, single-dataloader evaluation, and single-seed
training execution. Multi-split experiment orchestration lives in `mlp_workflow.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    import torch
    from torch import nn, optim
except ImportError:
    torch = None  # type: ignore
    nn = None  # type: ignore
    optim = None  # type: ignore

from src.config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    LEARNING_RATE,
    MAX_EPOCHS,
    WEIGHT_DECAY,
)
from src.graph_data import create_dataloaders
from src.mlp_model import BaselineMLP
from src.standardization import unstandardize_target_labels
from src.utils import get_device, set_seed


def train_one_epoch(
    model: Any,
    dataloader: Any,
    criterion: Any,
    optimizer: Any,
    device: Any,
) -> float:
    """Run one training epoch on the target GPU/CPU device and return average batch MSE loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    non_blocking = getattr(device, "type", None) == "cuda"

    for fp_batch, cell_batch, label_batch in dataloader:
        fp_batch = fp_batch.to(device, non_blocking=non_blocking)
        cell_batch = cell_batch.to(device, non_blocking=non_blocking)
        label_batch = label_batch.to(device, non_blocking=non_blocking)

        optimizer.zero_grad()
        predictions = model(fp_batch, cell_batch)
        loss = criterion(predictions, label_batch)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(1, n_batches)


def evaluate_model(
    model: Any,
    dataloader: Any,
    y_mean: float,
    y_std: float,
    device: Any,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluate a trained model on a DataLoader using the target GPU/CPU device, unstandardize
    predictions back to LN_IC50 log-micromolar units, and return (RMSE, Spearman, preds, trues).
    """
    model.eval()
    preds_list: list[np.ndarray] = []
    trues_list: list[np.ndarray] = []
    non_blocking = getattr(device, "type", None) == "cuda"

    with torch.no_grad():
        for fp_batch, cell_batch, label_batch in dataloader:
            fp_batch = fp_batch.to(device, non_blocking=non_blocking)
            cell_batch = cell_batch.to(device, non_blocking=non_blocking)
            outputs = model(fp_batch, cell_batch).detach().cpu().numpy().reshape(-1)
            preds_list.append(outputs)
            trues_list.append(label_batch.cpu().numpy().reshape(-1))

    preds = unstandardize_target_labels(np.concatenate(preds_list), y_mean, y_std)
    trues = unstandardize_target_labels(np.concatenate(trues_list), y_mean, y_std)

    rmse = float(np.sqrt(np.mean((preds - trues) ** 2)))
    sp = float(spearmanr(preds, trues).correlation)
    return rmse, sp, preds, trues


def train_single_seed(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int = 0,
    epochs: int = MAX_EPOCHS,
    patience: int = EARLY_STOPPING_PATIENCE,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    batch_size: int = BATCH_SIZE,
    device: Any | None = None,
    save_checkpoint_path: Path | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Train `BaselineMLP` on GPU (or fallback CPU) for a single random seed with early stopping
    on validation RMSE, restore the best checkpoint, and evaluate once on the held-out test set.
    """
    if torch is None:
        raise ImportError("PyTorch is required for train_single_seed().")

    set_seed(seed)
    if device is None:
        device = get_device()

    pin_memory = getattr(device, "type", None) == "cuda"
    train_loader, val_loader, test_loader, train_ds = create_dataloaders(
        train_df, val_df, test_df, batch_size=batch_size, pin_memory=pin_memory
    )

    cell_dim = len(train_ds.cell_cols)
    model = BaselineMLP(cell_dim=cell_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_rmse = float("inf")
    patience_counter = 0
    best_state: dict[str, Any] | None = None
    epochs_ran = 0

    for epoch in range(epochs):
        epochs_ran = epoch + 1
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_rmse, val_sp, _, _ = evaluate_model(
            model, val_loader, train_ds.y_mean, train_ds.y_std, device
        )

        if verbose:
            print(
                f"Epoch {epochs_ran:03d} | Val RMSE: {val_rmse:.4f} | Val Spearman: {val_sp:.4f}"
            )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            patience_counter = 0
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            if save_checkpoint_path is not None:
                torch.save(
                    {
                        "model_state": best_state,
                        "y_mean": train_ds.y_mean,
                        "y_std": train_ds.y_std,
                        "cell_cols": train_ds.cell_cols,
                    },
                    save_checkpoint_path,
                )
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping triggered at epoch {epochs_ran}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    test_rmse, test_sp, test_preds, test_trues = evaluate_model(
        model, test_loader, train_ds.y_mean, train_ds.y_std, device
    )

    return {
        "test_rmse": test_rmse,
        "test_spearman": test_sp,
        "test_preds": test_preds,
        "test_trues": test_trues,
        "n_epochs": epochs_ran,
        "cell_dim": cell_dim,
    }
