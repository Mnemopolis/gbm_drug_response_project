"""
Reusable synthetic data generators for unit testing data acquisition, deduplication,
splitting, standardization, pipeline auditing, baselines, uncertainty, and error analysis.
"""

from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd


def make_synthetic_merged_dataframe(n_cells: int = 8, n_drugs: int = 10, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic merged multi-omics + drug dataframe with known dimensions."""
    rng = np.random.default_rng(seed)
    rows = []
    drugs = [f"Compound_{i}" for i in range(n_drugs)] + ["Temozolomide", "Irinotecan"]
    scaffolds = [f"Scaffold_{i % 4}" for i in range(len(drugs))]

    for c_i in range(n_cells):
        cell_id = f"ACH-000{c_i:03d}"
        for d_i, (drug, scaf) in enumerate(zip(drugs, scaffolds)):
            row = {
                "ModelID": cell_id,
                "DRUG_NAME": drug,
                "DRUG_KEY": drug.upper().replace("_", ""),
                "SMILES": "c1ccccc1" if d_i % 2 == 0 else "c1ccncc1",
                "Scaffold": scaf,
                "Morgan_FP": rng.integers(0, 2, size=2048).astype(np.float32),
                "LN_IC50": float(d_i * 0.5 - c_i * 0.15 + rng.normal(0, 0.1)),
                "MGMT_Expr": float(rng.uniform(1.0, 4.0)),
            }
            for g_i in range(20):
                row[f"EXPR_G{g_i}"] = float(rng.normal(0, 1))
            for m_i in range(4):
                row[f"MUT_M{m_i}"] = float(rng.choice([0.0, 1.0]))
            rows.append(row)

    return pd.DataFrame(rows)


def write_synthetic_raw_csv_files(tmp_dir: Path) -> Dict[str, Path]:
    """Write miniature raw GDSC2, Model, Expression, Mutation, Methylation, and SMILES files."""
    paths = {
        "gdsc": tmp_dir / "GDSC2_fitted_dose_response_27Oct23.csv",
        "model": tmp_dir / "Model.csv",
        "expr": tmp_dir / "OmicsExpressionProteinCodingGenesTPMLogp1BatchCorrected.csv",
        "mut": tmp_dir / "OmicsSomaticMutationsMatrixHotspot.csv",
        "methyl_txt": tmp_dir / "CCLE_RRBS_TSS_1kb_20180614.txt",
        "methyl_csv": tmp_dir / "CCLE_RRBS_Methylation.csv",
        "smiles": tmp_dir / "drug_smiles.csv",
    }

    pd.DataFrame({
        "ModelID": ["ACH-000001", "ACH-000002", "ACH-000003", "ACH-000004"],
        "SangerModelID": ["SIDM001", "SIDM002", "SIDM003", "SIDM004"],
        "CellLineName": ["U87MG", "T98G", "LN229", "A172"],
        "OncotreeCode": ["GB", "GB", "GB", "GSARC"],
        "OncotreeSubtype": ["Glioblastoma", "Glioblastoma", "Glioblastoma", "Gliosarcoma"],
    }).to_csv(paths["model"], index=False)

    pd.DataFrame({
        "SANGER_MODEL_ID": ["SIDM001", "SIDM001", "SIDM002", "SIDM003", "SIDM004"],
        "CANCER_TYPE": ["Glioblastoma", "Glioblastoma", "Glioblastoma", "Glioblastoma", "Lung"],
        "DRUG_NAME": ["Temozolomide", "Temozolomide", "Temozolomide", "Irinotecan", "Temozolomide"],
        "LN_IC50": [2.0, 4.0, 3.5, 1.2, 5.0],
    }).to_csv(paths["gdsc"], index=False)

    pd.DataFrame(
        {
            "MGMT (4255)": [1.5, 2.5, 3.5, 0.5],
            "EGFR (1956)": [4.1, 3.2, 2.8, 5.0],
            "TP53 (7157)": [2.2, 1.8, 3.0, 2.7],
        },
        index=["ACH-000001", "ACH-000002", "ACH-000003", "ACH-000004"],
    ).to_csv(paths["expr"])

    pd.DataFrame({
        "ModelID": ["ACH-000001", "ACH-000002", "ACH-000003", "ACH-000001"],
        "IsDefaultEntryForModel": ["Yes", "Yes", "Yes", "No"],
        "SequencingID": ["S1", "S2", "S3", "S4"],
        "TP53": [1.0, 0.0, 1.0, 1.0],
        "PTEN": [0.0, 1.0, 0.0, 0.0],
        "ZERO_GENE": [0.0, 0.0, 0.0, 0.0],
    }).to_csv(paths["mut"], index=False)

    pd.DataFrame(
        {"U87MG": [0.85, 0.12], "T98G": [0.20, 0.45], "UNMAPPED": [0.50, 0.50]},
        index=["MGMT_10_131265000", "OTHER_TSS"],
    ).to_csv(paths["methyl_txt"], sep="\t")

    pd.DataFrame({
        "drug_name": ["Temozolomide", "Irinotecan", "Aspirin"],
        "canonical_smiles": [
            "CN1N=NC2=C(C(=O)N)N=CN2C1=O",
            "CCC1=C2CN3C(=CC4=C(C3=O)COC(=O)C4(CC)O)C2=NC5=C1C=C(C=C5)OC(=O)N6CCC(CC6)N7CCCCC7",
            "CC(=O)OC1=CC=CC=C1C(=O)O",
        ],
    }).to_csv(paths["smiles"], index=False)

    return paths
