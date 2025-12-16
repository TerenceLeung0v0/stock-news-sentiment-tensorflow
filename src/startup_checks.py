from typing import Iterable, Union
from pathlib import Path
from config import REQUIRED_DIRS, LABEL_MAP_FILE, TOKENIZER_FILE, BEST_MODEL_FILE, BEST_WEIGHT_FILE
from utils import is_file_non_empty

def ensure_required_dirs(dirs: Iterable[Union[str, Path]]) -> None:
    """
    Ensure required directories exist
    """
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def ensure_project_dirs() -> None:
    ensure_required_dirs(REQUIRED_DIRS)

def check_required_data(path: Path) -> None:
    if not is_file_non_empty(path):
        raise RuntimeError(f"Required data file is missing or empty: {path}")

def check_required_artifacts(
    require_label_map=True,
    require_tokenizer=True,
    require_model=True,
    require_weights=True
) -> None:
    """
    Validate all required artifacts before running notebook
    - require_label_map: Whether "label_map.json" must exist
    - require_tokenizer: Whether "tokenizer.json" must exist
    - require_model: Whether "bilstm_best_model.keras" must exist
    - require_weights: Whether "bilstm_best_weight.weights.h5" must exist
    Remark:
    - Inference typically requires either model file or weights file (depending on loading strategy)
    """
    missings = []
    
    if require_label_map and not is_file_non_empty(LABEL_MAP_FILE):
        missings.append(f"{LABEL_MAP_FILE} is missing or empty")
    
    if require_tokenizer and not is_file_non_empty(TOKENIZER_FILE):
        missings.append(f"{TOKENIZER_FILE} is missing or empty")
        
    if require_model and not is_file_non_empty(BEST_MODEL_FILE):
        missings.append(f"{BEST_MODEL_FILE} is missing or empty")

    if require_weights and not is_file_non_empty(BEST_WEIGHT_FILE):
        missings.append(f"{BEST_WEIGHT_FILE} is missing or empty")
        
    if missings:
        print("Some required artifacts are missing.")
        
        for missing in missings:
            print(missing)
        
        raise RuntimeError("Required artifacts are missing. Please generate them in prior notebooks")
    else:
        print("All required artifacts are found. Notebook is ready")
