from pathlib import Path

BASE_PATH = Path(__file__).resolve().parents[1]

# Data directories
DATA_DIR = BASE_PATH / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

DATA_OUTPUT_DIRS = [
    DATA_DIR,
    DATA_PROCESSED_DIR
]

# Dataset
TEXT_COL = "text"
LABEL_COL = "label"
DATASET_FILENAME = "financial_phrasebank.csv"   # Raw dataset filename
name = Path(DATASET_FILENAME).stem
ext = Path(DATASET_FILENAME).suffix
DATASET_CLEAN_FILENAME = f"{name}_clean{ext}"  # Cleaned dataset filename
RAW_FILE = DATA_RAW_DIR / DATASET_FILENAME
CLEAN_FILE = DATA_PROCESSED_DIR /  DATASET_CLEAN_FILENAME

# Artifacts directories
ARTIFACTS_DIR = BASE_PATH / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
PREPROCESSING_DIR = ARTIFACTS_DIR / "preprocessing"
RESULTS_DIR = ARTIFACTS_DIR / "results"
TUNING_MODELS_DIR = MODELS_DIR / "tuning"
BEST_MODELS_DIR = MODELS_DIR / "best" 

ARTIFACTS_DIRS = [
    ARTIFACTS_DIR,
    MODELS_DIR,
    PREPROCESSING_DIR,
    RESULTS_DIR,
    TUNING_MODELS_DIR,
    BEST_MODELS_DIR    
]

# Artifacts files
LABEL_MAP_FILE = PREPROCESSING_DIR / "label_map.json"
TOKENIZER_FILE = PREPROCESSING_DIR / "tokenizer.json"
BEST_MODEL_FILE = BEST_MODELS_DIR / "bilstm_best_model.keras"
BEST_WEIGHT_FILE = BEST_MODELS_DIR / "bilstm_best_weight.weights.h5"
BEST_MODEL_METADATA = RESULTS_DIR / "best_model_metadata.json"
FINAL_METRICS_FILE = RESULTS_DIR / "final_metrics.json"

# Required directories exist before any execution
REQUIRED_DIRS = DATA_OUTPUT_DIRS + ARTIFACTS_DIRS

RANDOM_STATE = 42

MAX_LEN = 50        # 95% percentile = 42 -> Rounded to nearest bucket for padding efficiency
VOCAB_SIZE = 10000   # Vocabularies = ~9.5k (Relative small dataset, capture all)
OOV_TOKEN = "<OOV>"


