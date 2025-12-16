from typing import Any
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_bilstm_model(
    cfg: dict[str, Any],
    vocab_size: int,
    max_len: int,
    classes_qty: int,
    verbose: bool=False
) -> Sequential:
    embedding_dim = cfg.get("embedding_dim", 64)
    lstm_units = cfg.get("lstm_units", 32)
    dropout = cfg.get("dropout", 0.3)
    dense_units = cfg.get("dense_units", 64)
    dense_dropout = cfg.get("dense_dropout", 0.3)
    lr = cfg.get("lr", 1e-3)
    
    model = Sequential([
        Input(shape=(max_len,)),
        Embedding(vocab_size, embedding_dim),
        Bidirectional(LSTM(lstm_units, return_sequences=False)),
        Dropout(dropout),
        Dense(dense_units, activation="relu"),
        Dropout(dense_dropout),
        Dense(classes_qty, activation="softmax")        
    ])

    optimizer = Adam(learning_rate=lr)

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    ) 

    if verbose:
        for key, value in cfg.items():
            print(f"{key}: {value}")
    
    return model