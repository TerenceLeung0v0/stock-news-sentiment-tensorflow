from typing import Sequence, Any
from pathlib import Path
from config import RESULTS_DIR, LABEL_MAP_FILE, TOKENIZER_FILE, BEST_MODEL_METADATA, FINAL_METRICS_FILE
from utils import is_file_non_empty
from tensorflow.keras.preprocessing.text import tokenizer_from_json, Tokenizer

import json

def path_serializer(obj):
    """
    Path is not serializable in JSON: Path -> str
    """
    if isinstance(obj, Path):
        return str(obj)
    
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def save_label_artifacts(
    label_order: Sequence[str],
    label_to_id: dict[str, int],
    id_to_label: dict[int, str],
) -> None:
    """
    Save label mapping artifacts in label_map.json
    Prerequisite:
    - label_order
    - label_to_id
    - id_to_label
    Make sure the inputs are built from build_label_mappings()
    """
    expected_label_ids = list(range(len(label_order)))
    
    if list(label_to_id.keys()) != list(label_order):
        raise ValueError("label_to_id keys must match label_order exactly")
    
    if [label_to_id[label] for label in label_order] != expected_label_ids:
        raise ValueError("label_to_id must be in label_order order")
    
    if [id_to_label[id] for id in expected_label_ids] != list(label_order):
        raise ValueError("id_to_label must invert label_to_id exactly")
    
    payload = {
        "label_order": list(label_order),
        "label_to_id": label_to_id,
        "id_to_label": id_to_label
    }
    
    LABEL_MAP_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    with open(LABEL_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False)
    
    print(f"Label artifacts are saved in {LABEL_MAP_FILE}")

def load_label_artifacts() -> tuple[list[str], dict[str, int], dict[int, str]]:
    if not is_file_non_empty(LABEL_MAP_FILE): 
        raise RuntimeError(f"Missing or empty label map: {LABEL_MAP_FILE}")
        
    with open(LABEL_MAP_FILE, "r", encoding="utf-8") as f:
        payload = json.load(f)
    
    if "label_order" not in payload or "label_to_id" not in payload or "id_to_label" not in payload:
        raise ValueError("Invalid: label_order, label_to_id or id_to_label are missing")
    
    label_order = payload["label_order"]
    label_to_id = payload["label_to_id"]
    id_to_label_raw = payload["id_to_label"]    # integers are converted into string in json'
    
    id_to_label = {int(k): v for k, v in id_to_label_raw.items()}
    
    return label_order, label_to_id, id_to_label

def save_tokenizer(tokenizer: Any) -> None:
    tokenizer_json = tokenizer.to_json()

    TOKENIZER_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    with open(TOKENIZER_FILE, "w", encoding="utf-8") as f:
        f.write(tokenizer_json)
    
    print(f"Tokenizer is saved to {TOKENIZER_FILE}")

def load_tokenizer() -> Tokenizer:
    with open(TOKENIZER_FILE, "r", encoding="utf-8") as f:
        text = f.read()
    
    return tokenizer_from_json(text)
        
def save_model_metadata(
    hyperparams: dict[str, Any],
    callbacks: list[dict[str, Any]],
    metrics: dict[str, Any],
    path: Path
) -> None:
    payload = {
        "hyperparams": hyperparams,
        "callbacks": callbacks,
        "metrics": metrics,
    }
    
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False, default=path_serializer)
    
    print(f"Model parameters are saved in {path}")


def load_model_metadata(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    
    required = {"hyperparams", "callbacks", "metrics"}
    missing = required - set(payload.keys())
    
    if missing:
        raise ValueError(f"Required keys are missing from the payload: {sorted(missing)}")
    
    return payload

def save_best_model_metadata(
    hyperparams: dict[str, Any],
    callbacks: list[dict[str, Any]],
    metrics: dict[str, Any]    
) -> None:
    save_model_metadata(
        hyperparams,
        callbacks,
        metrics,
        path=BEST_MODEL_METADATA,
    )

def load_best_model_metadata():
    return load_model_metadata(BEST_MODEL_METADATA)

def save_metrics(
    metrics: dict[str, Any],
    path: Path,
) -> None:
    """
    Save metrics to JSON file to artifacts/results
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False, default=path_serializer)
        
    print(f"Metrics are saved in {path}")

def save_latest_metrics(
    metrics: dict[str, Any],
    filename: str
) -> None:
    """
    Save the metrics rather than final metrics 
    """
    save_metrics(
        metrics=metrics,
        path=RESULTS_DIR / f"{filename}.json"
    )

def save_final_metrics(
    metrics: dict[str, Any],
) -> None:
    """
    Save final metrics 
    """    
    save_metrics(
        metrics=metrics,
        path=FINAL_METRICS_FILE
    )


