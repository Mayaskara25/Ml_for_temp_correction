"""
features_v2.py
---------------
Shared data-loading and feature engineering functions for the v2 multi-model
sensor correction comparison. Imported by train_multimodel_v2.py and
model_comparison_report_v2.py.
"""

import numpy as np
import pandas as pd

T_MIN = 20.0
T_MAX = 80.0
LOOKBACK = 10          # number of past K-type samples in the window
HADAMARD_WINDOW = 8    # power-of-2 sub-window length for the Hadamard branch


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load dataset.csv, sort by Timestamp_ms, drop duplicate timestamps and NaNs.
    Returns DataFrame with columns Timestamp_ms, K_Type_Temp_C, PT100_Temp_C, Error_C."""
    df = pd.read_csv(csv_path)

    n_before = len(df)
    df = df.sort_values('Timestamp_ms').reset_index(drop=True)

    n_before_dedup = len(df)
    df = df.drop_duplicates(subset='Timestamp_ms', keep='first').reset_index(drop=True)
    n_dropped_dupes = n_before_dedup - len(df)
    if n_dropped_dupes > 0:
        print(f"load_dataset: dropped {n_dropped_dupes} duplicate-timestamp rows")

    n_before_nan = len(df)
    df = df.dropna(subset=['K_Type_Temp_C', 'PT100_Temp_C']).reset_index(drop=True)
    n_dropped_nan = n_before_nan - len(df)
    if n_dropped_nan > 0:
        print(f"load_dataset: dropped {n_dropped_nan} rows with NaN in K_Type_Temp_C/PT100_Temp_C")

    print(f"load_dataset: {n_before} rows read, {len(df)} rows retained after cleaning")
    return df


def scale_temp(t, t_min=T_MIN, t_max=T_MAX):
    return (t - t_min) / (t_max - t_min)


def unscale_temp(s, t_min=T_MIN, t_max=T_MAX):
    return s * (t_max - t_min) + t_min


def compute_derivative(k_values: np.ndarray, dt_seconds: float) -> np.ndarray:
    """First-difference derivative dK/dt. First element set to 0 (no prior sample)."""
    d = np.zeros_like(k_values)
    d[1:] = (k_values[1:] - k_values[:-1]) / dt_seconds
    return d


def get_derivative_bounds(deriv_values: np.ndarray) -> tuple:
    """Return (min, max) of the derivative array, used to scale it to [0,1]
    the same way temperature is scaled. MUST be printed by the caller and
    hardcoded into the firmware — do not recompute on-device."""
    return float(deriv_values.min()), float(deriv_values.max())


def scale_derivative(d, d_min, d_max):
    # guard against d_min == d_max (flat data) by returning zeros in that case
    if d_max - d_min < 1e-9:
        return np.zeros_like(d)
    return (d - d_min) / (d_max - d_min)


def build_windows(k_scaled: np.ndarray, dk_scaled: np.ndarray, target_scaled: np.ndarray,
                   lookback: int = LOOKBACK):
    """Build sliding windows for Dense/TCN models.
    Returns:
      X_flat: shape (N, lookback*2) -- [k_0..k_{L-1}, dk_0..dk_{L-1}] flattened, for Dense
      X_seq:  shape (N, lookback, 2) -- [k, dk] per timestep, for TCN (channel-last)
      y:      shape (N, 1) -- target_scaled[i+lookback]
    where N = len(k_scaled) - lookback."""
    n = len(k_scaled) - lookback
    X_flat = np.zeros((n, lookback * 2), dtype=np.float32)
    X_seq = np.zeros((n, lookback, 2), dtype=np.float32)
    y = np.zeros((n, 1), dtype=np.float32)

    for i in range(n):
        k_win = k_scaled[i:i + lookback]
        dk_win = dk_scaled[i:i + lookback]
        X_flat[i] = np.concatenate([k_win, dk_win])
        X_seq[i, :, 0] = k_win
        X_seq[i, :, 1] = dk_win
        y[i, 0] = target_scaled[i + lookback]

    return X_flat, X_seq, y


def train_val_split_timeordered(X, y, val_fraction=0.1):
    """Time-ordered split -- NOT random shuffle. Take the last val_fraction of
    rows (in original time order) as validation. Random shuffling here would
    leak adjacent overlapping windows between train and val and give
    artificially good validation numbers."""
    n = len(X)
    n_val = int(round(n * val_fraction))
    n_train = n - n_val
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]
    return X_train, X_val, y_train, y_val
