"""
Feature Engineering for GNSS Spoofing Detection
================================================
Computes derived temporal features from raw GNSS observations:
  - First-order temporal differences (variation)
  - Rolling statistics (mean, std over a short window)
  - Position change magnitude
  - Signal quality change
"""
import numpy as np
import pandas as pd
from typing import List


TEMPORAL_DIFF_FEATURES = [
    "pseudorange",
    "carrier_phase",
    "cn0",
    "doppler",
    "elevation",
    "azimuth",
    "latitude",
    "longitude",
    "altitude",
    "velocity",
]

ROLLING_WINDOW = 5  # observations


def compute_temporal_differences(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Compute first-order temporal differences for numeric GNSS features.
    diff_X = X(t) - X(t-1)
    """
    for col in columns:
        if col in df.columns:
            df[f"diff_{col}"] = df[col].diff().fillna(0.0)
    return df


def compute_rolling_statistics(
    df: pd.DataFrame, columns: List[str], window: int = ROLLING_WINDOW
) -> pd.DataFrame:
    """
    Compute rolling mean and rolling std for key GNSS features.
    Captures short-term statistical behavior changes.
    """
    for col in columns:
        if col in df.columns:
            df[f"roll_mean_{col}"] = (
                df[col].rolling(window=window, min_periods=1).mean()
            )
            df[f"roll_std_{col}"] = (
                df[col].rolling(window=window, min_periods=1).std().fillna(0.0)
            )
    return df


def compute_position_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Euclidean-like magnitude of position change between consecutive observations.
    Large sudden jumps are a strong spoofing indicator.
    """
    if all(c in df.columns for c in ["latitude", "longitude", "altitude"]):
        dlat = df["latitude"].diff().fillna(0.0)
        dlon = df["longitude"].diff().fillna(0.0)
        dalt = df["altitude"].diff().fillna(0.0)
        # Approximate meters: 1 degree lat/lon ≈ 111,000 m
        df["position_change_m"] = np.sqrt(
            (dlat * 111_000) ** 2
            + (dlon * 111_000 * np.cos(np.radians(df.get("latitude", 0)))) ** 2
            + dalt ** 2
        )
    return df


def compute_pseudorange_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pseudorange rate = diff(pseudorange) / diff(time).
    Spoofed signals often show inconsistent pseudorange rates.
    """
    if "pseudorange" in df.columns:
        if "timestamp" in df.columns:
            dt = df["timestamp"].diff().replace(0, 1).fillna(1.0)
            df["pseudorange_rate"] = df["pseudorange"].diff().fillna(0.0) / dt
        else:
            df["pseudorange_rate"] = df["pseudorange"].diff().fillna(0.0)
    return df


def compute_cn0_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    """
    C/N0 anomaly = deviation from rolling mean.
    Significant drops below rolling mean indicate spoofing.
    """
    if "cn0" in df.columns:
        roll_mean = df["cn0"].rolling(window=10, min_periods=1).mean()
        df["cn0_anomaly"] = df["cn0"] - roll_mean
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps in sequence.
    Returns DataFrame with original + engineered features.
    """
    df = df.copy()

    # Sort by timestamp if available
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    # Compute all engineered features
    df = compute_temporal_differences(df, TEMPORAL_DIFF_FEATURES)
    df = compute_rolling_statistics(df, ["pseudorange", "cn0", "doppler", "carrier_phase"])
    df = compute_position_change(df)
    df = compute_pseudorange_rate(df)
    df = compute_cn0_anomaly(df)

    return df


def get_feature_columns(df: pd.DataFrame, exclude_cols: List[str] = None) -> List[str]:
    """
    Return all numeric feature column names (excluding label, timestamp, id, etc.)
    """
    if exclude_cols is None:
        exclude_cols = ["label", "timestamp", "id"]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in exclude_cols]
