"""Unit tests exclusively for `src.fingerprints`."""

import tempfile
import unittest
from pathlib import Path

try:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

from src.fingerprints import (
    build_drug_feature_table,
    compute_max_tanimoto_similarity,
    compute_scaffold_morgan_fp,
    extract_murcko_scaffold_smiles,
)
from tests.fixtures.synthetic_data import write_synthetic_raw_csv_files


class TestFingerprints(unittest.TestCase):
    """Verify RDKit Morgan fingerprint generation, Murcko scaffolds, and Tanimoto similarity."""

    def test_extract_murcko_scaffold_handles_none(self) -> None:
        self.assertIsNone(extract_murcko_scaffold_smiles(None))

    @unittest.skipUnless(RDKIT_AVAILABLE, "RDKit is not installed in this environment")
    def test_build_drug_feature_table_and_tanimoto(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_synthetic_raw_csv_files(Path(tmp))
            drug_table = build_drug_feature_table(paths["smiles"])
            self.assertEqual(len(drug_table), 3)
            self.assertEqual(drug_table["Morgan_FP"].iloc[0].shape, (2048,))

            morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
            fp1 = compute_scaffold_morgan_fp("c1ccccc1", morgan_gen)
            fp2 = compute_scaffold_morgan_fp("c1ccncc1", morgan_gen)
            sim = compute_max_tanimoto_similarity(fp1, [fp1, fp2])
            self.assertAlmostEqual(sim, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
