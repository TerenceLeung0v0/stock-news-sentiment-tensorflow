from typing import Sequence, Optional, Hashable
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from pathlib import Path
from config import RANDOM_STATE

import shutil
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import tensorflow as tf

def set_seed(seed: int=RANDOM_STATE) -> None:
    """
    Set random seed for reproducibility across Python, Numpy and TensorFlow
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)    

def reset_tf_state(seed: Optional[int]=None) -> None:
    """
    Clear TensorFlow session and set random seed optionally
    """
    tf.keras.backend.clear_session()
        
    if seed is not None:
        set_seed(seed)

def clear_folder(folder: str | Path, verbose=False) -> None: 
    folder_path = Path(folder)
    
    if not folder_path.exists():
        raise RuntimeError(f"{folder_path} does not exist")
    
    if not folder_path.is_dir():
        raise RuntimeError(f"{folder_path} is not a directory")
    
    for item in folder_path.iterdir():
        if item.is_file():
            item.unlink()
            if verbose: print(f"File: {item} is removed")
        elif item.is_dir():
            shutil.rmtree(item)
            if verbose: print(f"Folder: {item} is removed")

def is_file_non_empty(path: str | Path) -> bool:
    """
    Check if input path
    - exists
    - is a file
    - is non-empty
    """
    p = Path(path)
    
    return p.exists() and p.is_file() and p.stat().st_size > 0

def generate_config_name(cfg: dict, fields: Optional[list[str]]=None, prefix: str="h") -> str:
    """
    Generate config name based on hyperparameters/parameters inside
    - cfg:      dictionary containing hyperparameters/parameters, e.g. {"embedding_dim": 64, "dropout": 0.3, ...}
    - fields:   string list for what hyperparameters/parameters are included in the naming
    - prefix:   distinguish the config type, default "h" for hyperparameters
    Field order is preserved by the 'fields' argument to ensure deterministic naming
    """
    def is_tenth_step(value: float) -> bool:
        """
        Check for 0.1, 0.2, ... , 0.9
        """
        if value <= 0.0 or value >= 1.0:
            return False
        
        eps = 1e-8
        scaled = value * 10
        
        return abs(scaled - round(scaled)) < eps
    
    def normalize_value(value: int|float) -> str:
        if isinstance(value, float):
            if is_tenth_step(value):    # Special handling for 0.1, 0.2, ..., 0.9 -> 01, 02, ... , 09
                return str(value).replace(".", "")

            return f"{value:.0e}".replace("e-0", "e-").replace("e+0", "e")

        return str(value)   # int
    
    if fields is None:
        fields = list(cfg.keys())
    
    fragments = []
    common_keys = [field for field in fields if field in cfg]
    
    if common_keys:
        for common_key in common_keys:
            key = common_key[:3]    # Abbreviate to 1st 3 characters
            key_value = normalize_value(cfg[common_key])   # Normalize value to better expression in file name
            fragments.append(f"{key}{key_value}")
    
    return prefix + "_" + "_".join(fragments)

def build_label_mappings(labels: Sequence[str]) -> tuple[list[str], dict[str, int], dict[int, str]]:
    """
    Build label mappings from a sequence of labels
    - label_order: Preserve input order
    - label_to_id: dict of {label: id}
    - id_to_label: dict of {id: label}
    """
    label_order = list(pd.unique(labels))
    label_to_id = {label: idx for idx, label in enumerate(label_order)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    return label_order, label_to_id, id_to_label
    
def encode_labels(
    labels: Sequence[str],
    label_to_id: dict[str, int]
) -> np.ndarray:
    return np.array([label_to_id[label] for label in labels], dtype="int32")    # int32 for TensorFlow

def decode_labels(
    ids: Sequence[int],
    id_to_label: dict[int, str]
) -> list[str]:
    return [id_to_label[id] for id in ids]

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Sequence[Hashable],
    normalize: Optional[str]=None,
    title: str="Confusion Matrix",
    display_labels: Optional[Sequence[str]]=None
) -> None:
    if normalize is None:
        fmt = "d"
    else:
        fmt = ".2f"
    
    if display_labels is None:
        tick_labels = [str(label).capitalize() for label in labels]
    else:
        tick_labels = [str(display_label).capitalize() for display_label in display_labels]
    
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=tick_labels,
        yticklabels=tick_labels,
    )
    
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
def evaluate_and_plot(
    model,
    X: np.ndarray,
    y: np.ndarray,
    label_order: Sequence[str],
    split_name: str,
    title: str,
    is_show_confusion_matrix: bool=True,
    is_show_classification_report: bool=True,
    cm_normalize: Optional[str]="true"
) -> dict[str, float]:
    loss, acc = model.evaluate(X, y, verbose=0)
    y_pred = model.predict(X, verbose=0).argmax(axis=1)
    f1 = f1_score(y, y_pred, average="macro")
    
    metrics = {
        f"{split_name}_macro_f1": float(f1),
        f"{split_name}_acc": float(acc),
        f"{split_name}_loss": float(loss),
    }
    
    print("----------------- Metrics -----------------")
    df = pd.DataFrame([metrics]).sort_values(by=f"{split_name}_macro_f1", ascending=False)
    print(df.to_string(index=False))
    print("-------------------------------------------\n")
    
    labels_ids = list(range(len(label_order)))
    
    if is_show_classification_report:
        print(classification_report(
            y, y_pred,
            labels=labels_ids,
            target_names=list(label_order),
            zero_division=0
        ))
        
    if is_show_confusion_matrix:
        plot_confusion_matrix(
            y, y_pred,
            labels=labels_ids,
            normalize=cm_normalize,
            title=title,
            display_labels=label_order
        )
    
    return metrics

def show_label_distribution(
    y: np.ndarray,
    split_name: str,
) -> None:
    ps = pd.Series(y).value_counts(normalize=True).mul(100).round(2)
    
    print(f"{split_name} label distribution (%)")
    print(ps)
