"""
Molecular featurization module (Morgan Fingerprints & Bemis-Murcko Scaffolds).

Single Responsibility Principle (SRP):
Encapsulates all RDKit chemical structure parsing, 2048-bit Morgan fingerprint
generation, Bemis-Murcko scaffold extraction, and Tanimoto similarity computation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import DataStructs, rdFingerprintGenerator
    from rdkit.Chem.Scaffolds import MurckoScaffold
except ImportError:
    Chem = None  # type: ignore
    DataStructs = None  # type: ignore
    rdFingerprintGenerator = None  # type: ignore
    MurckoScaffold = None  # type: ignore

from src.config import MORGAN_FP_SIZE, MORGAN_RADIUS, SMILES_FILE
from src.deduplication import deduplicate_drug_smiles_table


def extract_murcko_scaffold_smiles(mol: Any) -> str | None:
    """Extract the canonical Bemis-Murcko scaffold SMILES string from an RDKit Mol object."""
    if mol is None or MurckoScaffold is None:
        return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        return None


def build_drug_feature_table(
    smiles_file: Path = SMILES_FILE,
    radius: int = MORGAN_RADIUS,
    fp_size: int = MORGAN_FP_SIZE,
) -> pd.DataFrame:
    """
    Load and deduplicate drug SMILES records, generate 2048-bit Morgan fingerprints,
    and extract Bemis-Murcko chemical scaffolds for OOD splitting.
    """
    if Chem is None or rdFingerprintGenerator is None:
        raise ImportError("RDKit is required for build_drug_feature_table().")

    drug_raw = pd.read_csv(smiles_file)
    drug_info = deduplicate_drug_smiles_table(drug_raw)

    morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=fp_size
    )
    drug_info["Mol"] = drug_info["SMILES"].map(Chem.MolFromSmiles)
    drug_info = drug_info[drug_info["Mol"].notna()].copy()

    drug_info["Morgan_FP"] = drug_info["Mol"].map(
        lambda mol: morgan_gen.GetFingerprintAsNumPy(mol).astype(np.float32)
    )
    drug_info["Scaffold"] = drug_info["Mol"].map(extract_murcko_scaffold_smiles)
    return drug_info[
        ["DRUG_NAME", "SMILES", "DRUG_KEY", "Morgan_FP", "Scaffold"]
    ].reset_index(drop=True)


def compute_scaffold_morgan_fp(smiles: Any, morgan_gen: Any) -> Any | None:
    """Compute an RDKit Morgan fingerprint bit vector for a molecule's Bemis-Murcko scaffold."""
    if Chem is None or MurckoScaffold is None:
        raise ImportError("RDKit is required for compute_scaffold_morgan_fp().")
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    scaf_smiles = extract_murcko_scaffold_smiles(mol) or ""
    scaf_mol = Chem.MolFromSmiles(scaf_smiles) if scaf_smiles else mol
    if scaf_mol is None:
        return None
    return morgan_gen.GetFingerprint(scaf_mol)


def compute_max_tanimoto_similarity(query_fp: Any, reference_fps: list[Any]) -> float:
    """Return the maximum Tanimoto similarity between `query_fp` and a list of RDKit fingerprints."""
    if DataStructs is None:
        raise ImportError("RDKit DataStructs is required for Tanimoto similarity.")
    if query_fp is None or len(reference_fps) == 0:
        return float("nan")
    similarities = DataStructs.BulkTanimotoSimilarity(query_fp, reference_fps)
    return float(np.max(similarities)) if len(similarities) > 0 else float("nan")
