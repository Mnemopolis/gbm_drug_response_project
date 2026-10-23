"""Unit tests exclusively for `src.config`."""

import unittest

from src.config import (
    ATOM_TO_IDX,
    ATOM_TYPES,
    DATA_DIR,
    GBM_DRUG_KEYWORDS,
    MODELS_DIR,
    MORGAN_FP_SIZE,
    PROJECT_ROOT,
    RESULTS_DIR,
    SPLIT_NAMES,
    get_split_data_paths,
)


class TestConfig(unittest.TestCase):
    """Verify centralized paths, constants, and split file path resolution."""

    def test_project_directories_exist(self) -> None:
        self.assertTrue(PROJECT_ROOT.exists())
        self.assertTrue(MODELS_DIR.exists())
        self.assertTrue(RESULTS_DIR.exists())
        self.assertEqual(DATA_DIR.parent, PROJECT_ROOT)

    def test_get_split_data_paths_valid_and_invalid(self) -> None:
        for split in SPLIT_NAMES:
            paths = get_split_data_paths(split)
            self.assertEqual(paths["train"].name, f"train_{split}.pkl")
            self.assertEqual(paths["test"].name, f"test_{split}.pkl")

        with self.assertRaises(ValueError):
            get_split_data_paths("invalid_split")

    def test_domain_constants(self) -> None:
        self.assertEqual(MORGAN_FP_SIZE, 2048)
        self.assertEqual(len(ATOM_TYPES), len(ATOM_TO_IDX))
        self.assertIn("Temozolomide", GBM_DRUG_KEYWORDS)


if __name__ == "__main__":
    unittest.main()
