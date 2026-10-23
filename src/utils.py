"""
General utility functions for reproducibility, GPU/CPU device management,
and string/column normalization.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterable
from typing import Any

import numpy as np

try:
    import torch
except ImportError:
    torch = None  # type: ignore


def set_seed(seed: int = 42) -> None:
    """Fix random seeds across Python, NumPy, and PyTorch (CPU & GPU) for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def get_device(prefer_device: str | None = None, require_gpu: bool = False) -> Any:
    """Select and return the PyTorch compute device, prioritizing GPU (CUDA or Apple MPS)."""
    if torch is None:
        raise ImportError(
            "PyTorch is required for get_device(). Activate your .vene environment."
        )

    if prefer_device is not None and prefer_device != "auto":
        return torch.device(prefer_device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    if require_gpu:
        raise RuntimeError(
            "GPU training was requested (--require-gpu), but neither CUDA nor MPS is available. "
            "Install CUDA-enabled PyTorch in .vene via:\n"
            "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
        )

    return torch.device("cpu")


def log_device_info(device: Any) -> None:
    """Print active hardware acceleration details so the user can verify GPU execution."""
    if torch is not None and getattr(device, "type", None) == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        print(f"[Device] Using NVIDIA CUDA GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    elif getattr(device, "type", None) == "mps":
        print("[Device] Using Apple Silicon MPS GPU Acceleration")
    else:
        print(
            "[Device] WARNING: Running on CPU (torch.cuda.is_available() == False).\n"
            "         To train on an NVIDIA GPU in .vene, install CUDA-enabled PyTorch:\n"
            "         pip install torch --index-url https://download.pytorch.org/whl/cu121"
        )


def normalize_drug_name(drug_name: str) -> str:
    """Normalize a drug name by upper-casing and stripping non-alphanumeric characters."""
    return re.sub(r"[^A-Z0-9]", "", str(drug_name).upper().strip())


def find_matching_column(
    candidates: Iterable[str], available_columns: Iterable[str]
) -> str:
    """Return the first candidate column name present in `available_columns`."""
    available_set: list[str] = list(available_columns)
    for candidate in candidates:
        if candidate in available_set:
            return candidate
    raise ValueError(
        f"Could not find any candidate column from {list(candidates)} in {available_set}."
    )
