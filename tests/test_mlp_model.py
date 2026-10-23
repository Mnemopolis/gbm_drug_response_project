"""Unit tests exclusively for `src.mlp_model`."""

import unittest

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

if TORCH_AVAILABLE:
    from src.mlp_model import BaselineMLP


class TestMlpModel(unittest.TestCase):
    """Verify Two-Tower BaselineMLP neural network architecture and forward pass shapes."""

    @unittest.skipUnless(
        TORCH_AVAILABLE, "PyTorch is not installed in this environment"
    )
    def test_baseline_mlp_forward_shape(self) -> None:
        model = BaselineMLP(fp_dim=2048, cell_dim=16, hidden_dim=64, dropout=0.2)
        fp = torch.randn(10, 2048)
        cell = torch.randn(10, 16)
        out = model(fp, cell)
        self.assertEqual(out.shape, (10,))


if __name__ == "__main__":
    unittest.main()
