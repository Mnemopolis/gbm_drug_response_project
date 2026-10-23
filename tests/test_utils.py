"""Unit tests exclusively for `src.utils`."""

import unittest
import numpy as np
from src.utils import find_matching_column, normalize_drug_name, set_seed


class TestUtils(unittest.TestCase):
    """Verify deterministic random seeding, drug string normalization, and column matching."""

    def test_set_seed_reproducibility(self) -> None:
        set_seed(123)
        a = np.random.rand(5)
        set_seed(123)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_normalize_drug_name(self) -> None:
        self.assertEqual(normalize_drug_name("N-acetyl cysteine"), "NACETYLCYSTEINE")
        self.assertEqual(normalize_drug_name("  gsk-2830371a "), "GSK2830371A")

    def test_find_matching_column(self) -> None:
        self.assertEqual(
            find_matching_column(["canonical_smiles", "SMILES"], ["id", "SMILES"]),
            "SMILES",
        )
        with self.assertRaises(ValueError):
            find_matching_column(["missing"], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
