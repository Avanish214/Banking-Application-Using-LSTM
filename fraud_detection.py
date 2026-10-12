"""Data preparation and LSTM training for chronological fraud sequences."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class PreparedData:
    train_x: np.ndarray
    train_y: np.ndarray
    validation_x: np.ndarray
    validation_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    test_amounts: np.ndarray
    test_times: np.ndarray | None
    feature_names: list[str]
    scaler: StandardScaler


def prepare_sequences(frame: pd.DataFrame, sequence_length: int = 5) -> PreparedData:
    """Create leakage-conscious, chronological windows from a labeled CSV."""
    if "Class" not in frame.columns:
        raise ValueError("The uploaded CSV must contain a 'Class' target column.")
    if sequence_length < 2:
        raise ValueError("Sequence length must be at least 2 transactions.")

    data = frame.copy()
    if "Time" in data.columns:
        data = data.sort_values("Time", kind="stable")
    data = data.reset_index(drop=True)

    labels = pd.to_numeric(data.pop("Class"), errors="coerce").to_numpy()
    if np.isnan(labels).any() or not np.isin(labels, [0, 1]).all():
        raise ValueError("'Class' must contain only binary labels 0 and 1.")
    if not data.shape[1]:
        raise ValueError("The CSV must include at least one numeric transaction feature.")
    non_numeric = data.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(f"All input features must be numeric. Non-numeric: {', '.join(non_numeric)}")
    if not np.isfinite(data.to_numpy(dtype=np.float64)).all():
        raise ValueError("Transaction features contain missing or infinite values.")

    row_count = len(data)
    train_end = int(row_count * 0.70)
    validation_end = int(row_count * 0.85)
    if row_count < sequence_length * 3 or min(train_end, validation_end - train_end, row_count - validation_end) <= sequence_length:
        raise ValueError("The CSV is too small for the selected sequence length and chronological splits.")

    feature_names = data.columns.tolist()
    raw_features = data.to_numpy(dtype=np.float32)
    scaler = StandardScaler()
    scaler.fit(raw_features[:train_end])
    scaled = scaler.transform(raw_features).astype(np.float32)

    def make_windows(start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
        segment = scaled[start:end]
        windows = np.lib.stride_tricks.sliding_window_view(
            segment, window_shape=sequence_length, axis=0
        ).transpose(0, 2, 1)
        inputs = np.ascontiguousarray(windows, dtype=np.float32)
        targets = labels[start + sequence_length - 1:end].astype(np.float32)
        return inputs, targets

    train_x, train_y = make_windows(0, train_end)
    validation_x, validation_y = make_windows(train_end, validation_end)
    test_x, test_y = make_windows(validation_end, row_count)

    train_counts = np.bincount(train_y.astype(np.int64), minlength=2)
    if (train_counts == 0).any():
        raise ValueError("The training period must contain both genuine and fraudulent transactions.")
    test_start = validation_end + sequence_length - 1
    test_amounts = data["Amount"].to_numpy()[test_start:] if "Amount" in data else np.full(len(test_y), np.nan)
    test_times = data["Time"].to_numpy()[test_start:] if "Time" in data else None

    return PreparedData(
        train_x=train_x,
        train_y=train_y,
        validation_x=validation_x,
        validation_y=validation_y,
        test_x=test_x,
        test_y=test_y,
        test_amounts=test_amounts,
        test_times=test_times,
        feature_names=feature_names,
        scaler=scaler,
    )


def train_lstm(prepared: PreparedData, epochs: int = 10, batch_size: int = 512):
    """Fit the requested two-layer LSTM, weighting the minority fraud class."""
    from tensorflow import keras

    inputs = keras.Input(shape=prepared.train_x.shape[1:])
    hidden = keras.layers.LSTM(64, return_sequences=True)(inputs)
    hidden = keras.layers.Dropout(0.3)(hidden)
    hidden = keras.layers.LSTM(32)(hidden)
    hidden = keras.layers.Dense(16, activation="relu")(hidden)
    outputs = keras.layers.Dense(1, activation="sigmoid")(hidden)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="roc_auc"),
        ],
    )

    counts = np.bincount(prepared.train_y.astype(np.int64), minlength=2)
    total = len(prepared.train_y)
    class_weights = {label: total / (2 * count) for label, count in enumerate(counts)}
    history = model.fit(
        prepared.train_x,
        prepared.train_y,
        validation_data=(prepared.validation_x, prepared.validation_y),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=[keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)],
        verbose=0,
    )
    probabilities = model.predict(prepared.test_x, batch_size=batch_size, verbose=0).ravel()
    return model, history, probabilities