"""
train_dense_esp32.py
--------------------
Train and export a small Dense model for ESP32 temperature correction.

The default model stays intentionally simple: a flat lookback window of K-type
temperatures maps to a same-time PT100/reference target. Dense ops are broadly
supported by TensorFlow Lite Micro on generic ESP32 boards, unlike LSTM kernels.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


CSV_FILE = "synthetic_training_data.csv"
HEADER_FILE = "model_data_candidate.h"
LOOKBACK = 10
T_MIN = 20.0
T_MAX = 80.0


@dataclass(frozen=True)
class SplitData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    target_train_c: np.ndarray
    target_val_c: np.ndarray
    target_test_c: np.ndarray
    k_train_c: np.ndarray
    k_val_c: np.ndarray
    k_test_c: np.ndarray


def scale(temp: np.ndarray | float) -> np.ndarray | float:
    return (temp - T_MIN) / (T_MAX - T_MIN)


def unscale_temp(scaled: np.ndarray | float) -> np.ndarray | float:
    return scaled * (T_MAX - T_MIN) + T_MIN


def load_training_frame(csv_file: str, source: str | None, allow_mixed_sources: bool) -> pd.DataFrame:
    df = pd.read_csv(csv_file)
    df = normalize_training_columns(df)

    if "Source" in df.columns:
        sources = sorted(str(item) for item in df["Source"].dropna().unique())
        if source is not None:
            df = df[df["Source"].astype(str) == source].copy()
            if df.empty:
                raise ValueError(f"No rows found for Source={source!r}; available sources: {sources}")
        elif len(sources) > 1 and not allow_mixed_sources:
            raise ValueError(
                "Training data contains multiple Source values "
                f"{sources}. Pass --source or --allow-mixed-sources."
            )

    required = ["K_Type_Temp_C", "PT100_Temp_C"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required training columns: {missing}")

    df = df.dropna(subset=required).copy()
    if len(df) <= LOOKBACK:
        raise ValueError(f"Need more than {LOOKBACK} valid rows; found {len(df)}")
    return df


def normalize_training_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.copy()

    if "K_Type_Temp_C" not in renamed.columns:
        if "Live_K_Temp_C" in renamed.columns:
            renamed["K_Type_Temp_C"] = renamed["Live_K_Temp_C"]
        elif "Synthetic_K_Temp_C" in renamed.columns:
            renamed["K_Type_Temp_C"] = renamed["Synthetic_K_Temp_C"]

    if "PT100_Temp_C" not in renamed.columns:
        if "Live_PT100_Temp_C" in renamed.columns:
            renamed["PT100_Temp_C"] = renamed["Live_PT100_Temp_C"]
        elif "Synthetic_PT100_Temp_C" in renamed.columns:
            renamed["PT100_Temp_C"] = renamed["Synthetic_PT100_Temp_C"]

    if "K_Sensor_OK" in renamed.columns:
        renamed = renamed[(renamed["K_Sensor_OK"].isna()) | (renamed["K_Sensor_OK"].astype(str) != "0")]
    if "PT100_Sensor_OK" in renamed.columns:
        renamed = renamed[
            (renamed["PT100_Sensor_OK"].isna()) | (renamed["PT100_Sensor_OK"].astype(str) != "0")
        ]

    return renamed


def build_windows(
    df: pd.DataFrame,
    lookback: int,
    target_mode: str = "same-time",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if target_mode not in {"same-time", "forecast"}:
        raise ValueError("target_mode must be 'same-time' or 'forecast'")

    k_values = df["K_Type_Temp_C"].to_numpy(dtype=np.float32)
    pt_values = df["PT100_Temp_C"].to_numpy(dtype=np.float32)
    k_scaled = scale(k_values).astype(np.float32)
    pt_scaled = scale(pt_values).astype(np.float32)

    X, y, target_c, newest_k_c = [], [], [], []
    max_start = len(df) - lookback + 1
    if target_mode == "forecast":
        max_start -= 1

    for start in range(max_start):
        newest_idx = start + lookback - 1
        target_idx = newest_idx if target_mode == "same-time" else newest_idx + 1
        X.append(k_scaled[start : start + lookback])
        y.append(pt_scaled[target_idx])
        target_c.append(pt_values[target_idx])
        newest_k_c.append(k_values[newest_idx])

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.float32).reshape(-1, 1),
        np.asarray(target_c, dtype=np.float32),
        np.asarray(newest_k_c, dtype=np.float32),
    )


def chronological_split(
    X: np.ndarray,
    y: np.ndarray,
    target_c: np.ndarray,
    newest_k_c: np.ndarray,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
) -> SplitData:
    if not 0 < train_fraction < 1 or not 0 < val_fraction < 1:
        raise ValueError("Split fractions must be between 0 and 1")
    if train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction must leave a test split")

    n = len(X)
    train_end = int(n * train_fraction)
    val_end = train_end + int(n * val_fraction)
    if train_end == 0 or val_end <= train_end or val_end >= n:
        raise ValueError(f"Not enough samples for chronological split: {n}")

    return SplitData(
        X[:train_end],
        y[:train_end],
        X[train_end:val_end],
        y[train_end:val_end],
        X[val_end:],
        y[val_end:],
        target_c[:train_end],
        target_c[train_end:val_end],
        target_c[val_end:],
        newest_k_c[:train_end],
        newest_k_c[train_end:val_end],
        newest_k_c[val_end:],
    )


def regression_metrics(y_true_c: Iterable[float], y_pred_c: Iterable[float]) -> dict[str, float]:
    y_true = np.asarray(y_true_c, dtype=np.float64)
    y_pred = np.asarray(y_pred_c, dtype=np.float64)
    residual = y_pred - y_true
    return {
        "mae_c": float(np.mean(np.abs(residual))),
        "rmse_c": float(np.sqrt(np.mean(residual**2))),
        "bias_c": float(np.mean(residual)),
        "std_c": float(np.std(residual)),
    }


def fit_linear_baseline(x_train_c: np.ndarray, y_train_c: np.ndarray) -> np.poly1d:
    coeffs = np.polyfit(x_train_c, y_train_c, deg=1)
    return np.poly1d(coeffs)


def fit_polynomial_baseline(x_train_c: np.ndarray, y_train_c: np.ndarray) -> np.poly1d:
    coeffs = np.polyfit(x_train_c, y_train_c, deg=2)
    return np.poly1d(coeffs)


def build_dense_model(input_size: int):
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.models import Sequential

    model = Sequential(
        [
            Input(shape=(input_size,)),
            Dense(16, activation="relu"),
            Dense(8, activation="relu"),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def convert_to_tflite(model, X_train: np.ndarray, full_int8: bool) -> bytes:
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    if full_int8:
        def representative_dataset():
            for sample in X_train[: min(len(X_train), 256)]:
                yield [sample.reshape(1, -1).astype(np.float32)]

        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    return converter.convert()


def write_model_header(tflite_model: bytes, header_file: str, full_int8: bool) -> None:
    hex_parts = []
    for i, byte_val in enumerate(tflite_model):
        if i % 12 == 0:
            hex_parts.append("\n  ")
        hex_parts.append(f"0x{byte_val:02x}, ")

    quantization_note = (
        "Full integer int8 conversion with representative samples."
        if full_int8
        else "Optimize.DEFAULT conversion; weights may be compressed while activations may remain float."
    )
    header = f"""\
// Auto-generated by train_dense_esp32.py; do not edit manually.
// Dense MLP for generic ESP32 TensorFlow Lite Micro compatibility.
// {quantization_note}
#ifndef MODEL_DATA_H
#define MODEL_DATA_H

// 8-byte alignment required by TFLite Micro flatbuffer parser
alignas(8) const unsigned char model_data[] = {{{''.join(hex_parts)}
}};
const unsigned int model_data_len = {len(tflite_model)};

#endif // MODEL_DATA_H
"""
    Path(header_file).write_text(header)


def print_metrics(name: str, metrics: dict[str, float]) -> None:
    print(
        f"{name:>14}: MAE={metrics['mae_c']:.4f} C, "
        f"RMSE={metrics['rmse_c']:.4f} C, bias={metrics['bias_c']:+.4f} C, "
        f"std={metrics['std_c']:.4f} C"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ESP32 Dense temperature correction model.")
    parser.add_argument("--csv", default=CSV_FILE)
    parser.add_argument("--header-file", default=HEADER_FILE)
    parser.add_argument("--lookback", type=int, default=LOOKBACK)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--target-mode", choices=["same-time", "forecast"], default="same-time")
    parser.add_argument("--source", default=None)
    parser.add_argument("--allow-mixed-sources", action="store_true")
    parser.add_argument("--full-int8", action="store_true")
    parser.add_argument(
        "--no-write-header",
        action="store_true",
        help="Train and convert the model, but do not write a C header.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading {args.csv}...")
    df = load_training_frame(args.csv, args.source, args.allow_mixed_sources)

    X, y, target_c, newest_k_c = build_windows(df, args.lookback, args.target_mode)
    split = chronological_split(X, y, target_c, newest_k_c)

    print(f"Samples: train={len(split.X_train)}, val={len(split.X_val)}, test={len(split.X_test)}")
    print(
        "Target coverage: "
        f"train {split.target_train_c.min():.2f}-{split.target_train_c.max():.2f} C, "
        f"val {split.target_val_c.min():.2f}-{split.target_val_c.max():.2f} C, "
        f"test {split.target_test_c.min():.2f}-{split.target_test_c.max():.2f} C"
    )
    print(f"Target alignment: {args.target_mode}")

    print("\nBaselines on test split:")
    print_metrics("raw_k", regression_metrics(split.target_test_c, split.k_test_c))

    linear = fit_linear_baseline(split.k_train_c, split.target_train_c)
    print_metrics("linear", regression_metrics(split.target_test_c, linear(split.k_test_c)))

    polynomial = fit_polynomial_baseline(split.k_train_c, split.target_train_c)
    print_metrics("poly2", regression_metrics(split.target_test_c, polynomial(split.k_test_c)))

    print("\nTraining Dense MLP...")
    model = build_dense_model(split.X_train.shape[1])
    model.summary()
    model.fit(
        split.X_train,
        split.y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(split.X_val, split.y_val),
        verbose=1,
    )

    pred_test_c = unscale_temp(model.predict(split.X_test, verbose=0).reshape(-1))
    print("\nDense MLP on test split:")
    print_metrics("dense_mlp", regression_metrics(split.target_test_c, pred_test_c))

    print("\nConverting to TFLite...")
    tflite_model = convert_to_tflite(model, split.X_train, full_int8=args.full_int8)
    print(f"TFLite model size: {len(tflite_model)} bytes")
    print("Sampling budget: 500 ms; benchmark inference on ESP32 after upload.")

    if args.no_write_header:
        print("Header write skipped by --no-write-header.")
    else:
        print(f"Writing {args.header_file}...")
        write_model_header(tflite_model, args.header_file, full_int8=args.full_int8)


if __name__ == "__main__":
    main()
