from charset_normalizer import from_path
from config import TEXT_COL, LABEL_COL
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np
import pandas as pd

def normalize_label(label_series: pd.Series) -> pd.Series:
    """
    Normalize label strings:
    - lowercase
    - remove single quote
    - remove double quotes
    - trim whitespace
    """
    return (
        label_series.astype(str)
        .str.lower()
        .str.replace("'", "", regex=False)  # Remove single quote
        .str.replace('"', "", regex=False)  # Remove double quotes
        .str.strip()
    )

def detect_encoding(data_path: str) -> str:
    result = from_path(data_path).best()
    
    if result is None:
        raise ValueError("Unable to detect encoding")
    
    return result.encoding

def load_tabular_data(data_path: str, **kwargs) -> pd.DataFrame:
    """
    Load tabular file with suitable encodings
    """
    encoding = detect_encoding(data_path)
    
    return pd.read_csv(data_path, encoding=encoding, **kwargs)


def load_and_clean_data(data_path: str, basic_clean: bool=True, clean_label: bool=True) -> pd.DataFrame:
    """
    Load tabular file (.csv, .tsv, .psv etc) and apply basic cleaning
    - drop NaN values
    - drop empty text rows
    - drop empty label rows
    - optionally normalize label strings
    Note:
    This function normalizes the input schema to:
    - TEXT_COL = "text"
    - LABEL_COL = "label"
    regardless of original column names.
    """
    df = load_tabular_data(data_path, names=[TEXT_COL, LABEL_COL], header=0)
    
    # Basic cleaning
    if basic_clean:
        df = df.dropna(subset=[TEXT_COL, LABEL_COL])            # Drop NaN values
        df = df[df[TEXT_COL].str.strip() != ""]               # Drop empty text rows
        df = df[df[LABEL_COL].astype(str).str.strip() != ""]  # Symmetrically drop empty label rows
    
    # Normalize label strings
    if clean_label:
        df[LABEL_COL] = normalize_label(df[LABEL_COL])
    
    return df

def text_to_padded_sequences(
    tokenizer,
    texts:np.ndarray,
    max_len: int,
    padding: str="post",
    truncating: str="post"
) -> tuple[np.ndarray, list[list[int]]]:
    """
    Convert raw texts -> Sequences and Padded Sequences
    Raw sequences are returned for analysis/debugging and are not required for training
    """
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(
        sequences,
        maxlen=max_len,
        padding=padding,
        truncating=truncating
    )
    
    return padded, sequences
