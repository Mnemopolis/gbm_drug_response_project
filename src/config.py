"""
Centralized configuration for directory paths, raw data files, hyperparameters,
and Glioblastoma (GBM) pharmacological constants.
"""

from pathlib import Path

# 1. Directory Paths (anchored to project root via pathlib)
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "Data"
MODELS_DIR: Path = PROJECT_ROOT / "Models"
RESULTS_DIR: Path = PROJECT_ROOT / "Results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 2. Raw Input Data Paths
GDSC_FILE: Path = DATA_DIR / "GDSC2_fitted_dose_response_27Oct23.csv"
EXPR_FILE: Path = (
    DATA_DIR / "OmicsExpressionProteinCodingGenesTPMLogp1BatchCorrected.csv"
)
MUT_FILE: Path = DATA_DIR / "OmicsSomaticMutationsMatrixHotspot.csv"
MODEL_FILE: Path = DATA_DIR / "Model.csv"
METHYL_TXT_FILE: Path = DATA_DIR / "CCLE_RRBS_TSS_1kb_20180614.txt"
METHYL_CSV_FILE: Path = DATA_DIR / "CCLE_RRBS_Methylation.csv"
SMILES_FILE: Path = DATA_DIR / "drug_smiles.csv"

# 3. Cached Splits & Model Artifact Paths
SPLIT_NAMES: list[str] = ["random", "leave_cell", "scaffold"]
BEST_RANDOM_MODEL_PATH: Path = MODELS_DIR / "best_fp_random.pt"
OOD_RESULTS_PKL: Path = MODELS_DIR / "OOD_results.pkl"
OOD_PREDS_PKL: Path = MODELS_DIR / "OOD_preds.pkl"


def get_split_data_paths(split_name: str) -> dict[str, Path]:
    """Return the train and test pickle paths for a given split regime."""
    if split_name not in SPLIT_NAMES:
        raise ValueError(
            f"Unknown split_name '{split_name}'. Expected one of {SPLIT_NAMES}."
        )
    return {
        "train": MODELS_DIR / f"train_{split_name}.pkl",
        "test": MODELS_DIR / f"test_{split_name}.pkl",
    }


# 4. Experimental & Model Hyperparameters
DEFAULT_SEED: int = 42
TEST_SIZE: float = 0.20
VAL_SIZE: float = 0.15
N_PCA_COMPONENTS: int = 32

MORGAN_RADIUS: int = 2
MORGAN_FP_SIZE: int = 2048

HIDDEN_DIM: int = 256
DROPOUT_RATE: float = 0.30
BATCH_SIZE: int = 64
LEARNING_RATE: float = 1e-4
WEIGHT_DECAY: float = 1e-4
MAX_EPOCHS: int = 100
EARLY_STOPPING_PATIENCE: int = 15
N_MODEL_SEEDS: int = 5

# 5. Chemical & Clinical Domain Constants
ATOM_TYPES: list[str] = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si", "Se"]
ATOM_TO_IDX: dict[str, int] = {atom: idx for idx, atom in enumerate(ATOM_TYPES)}

GBM_DRUG_KEYWORDS: list[str] = [
    "Temozolomide",
    "TMZ",
    "Lomustine",
    "Carmustine",
    "EGFR",
    "Erlotinib",
    "Gefitinib",
    "Lapatinib",
    "BRAF",
    "Vemurafenib",
    "Dabrafenib",
    "MTOR",
    "Everolimus",
    "Temsirolimus",
    "PI3K",
    "Buparlisib",
    "Irinotecan",
    "Etoposide",
]

KEY_GBM_DRUGS_FOR_INSPECTION: list[str] = [
    "Temozolomide",
    "TMZ",
    "Lomustine",
    "Carmustine",
    "Irinotecan",
    "Etoposide",
]
