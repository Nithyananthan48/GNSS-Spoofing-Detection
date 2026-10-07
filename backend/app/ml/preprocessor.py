"""
GNSS Data Preprocessing Pipeline
===================================
Steps:
  1. Load CSV
  2. Validate required columns
  3. Handle missing values
  4. Remove duplicates
  5. Handle invalid numeric values
  6. Sort by timestamp if available
  7. Apply feature engineering
  8. Normalize/standardize features (scaler fitted only on train data)
  9. Encode labels
  10. Generate sequential time windows (sequences)
  11. Split into train / validation / test sets
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from app.ml.sklearn_compat import StandardScaler, train_test_split
import joblib

from app.config import (
    SEQUENCE_LENGTH,
    BIGRU_CONFIG,
    SCALER_PATH,
    FEATURE_CONFIG_PATH,
)
from app.ml.feature_engineering import engineer_features, get_feature_columns


COMMON_TARGET_COLUMNS = [
    "class", "target", "attack", "spoofing", "is_spoofed",
    "is_attack", "ground_truth", "spoofed", "status", "category"
]

VALID_NORMAL_VALUES = {0, 0.0, "0", "0.0", "normal", "Normal", "NORMAL", "authentic", "Authentic", "AUTHENTIC", False, "false", "False"}
VALID_SPOOFED_VALUES = {1, 1.0, "1", "1.0", "spoofed", "Spoofed", "SPOOFED", "attack", "Attack", "ATTACK", True, "true", "True"}


def detect_target_column_candidates(columns: List[str]) -> List[str]:
    """Detect potential alternative target column names in the dataset."""
    return [c for c in columns if c.lower() in COMMON_TARGET_COLUMNS and c != "label"]


def load_and_validate(
    filepath: str,
    require_label: bool = True,
    target_column: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load CSV and perform comprehensive validation:
      - Validates CSV syntax and non-empty shape
      - Detects and maps target_column if specified
      - Enforces strict binary label validation when required
      - Detects alternative column names if label is missing
    """
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        raise ValueError(f"Cannot read CSV file: {e}")

    if df.empty:
        raise ValueError("Uploaded dataset is empty (0 rows).")

    # If target_column is specified by user, validate and map it to 'label'
    if target_column:
        if target_column not in df.columns:
            raise ValueError(
                f"Specified target column '{target_column}' does not exist in dataset. "
                f"Available columns: {df.columns.tolist()}"
            )
        df = df.rename(columns={target_column: "label"})

    if require_label:
        if "label" not in df.columns:
            candidates = detect_target_column_candidates(df.columns.tolist())
            if candidates:
                raise ValueError(
                    f"Dataset is missing required 'label' column. "
                    f"Detected potential target column(s): {candidates}. "
                    f"Please map one of these columns to 'label'. "
                    f"Dataset shape: {len(df)} rows × {len(df.columns)} columns."
                )
            raise ValueError(
                f"Dataset must contain a 'label' column with binary values "
                f"(0 = Normal, 1 = Spoofed). "
                f"Available columns: {df.columns.tolist()}. "
                f"Dataset shape: {len(df)} rows × {len(df.columns)} columns."
            )

        # Validate label contents
        if df["label"].isnull().all():
            raise ValueError("The 'label' column contains only null/missing values.")

        unique_labels = df["label"].dropna().unique().tolist()
        invalid_labels = []
        for val in unique_labels:
            if val not in VALID_NORMAL_VALUES and val not in VALID_SPOOFED_VALUES:
                invalid_labels.append(str(val))

        if len(invalid_labels) > 0:
            raise ValueError(
                f"Invalid label value(s) detected: {invalid_labels[:10]}. "
                f"The 'label' column must contain binary values only: "
                f"0 (Normal) and 1 (Spoofed). Real user labels will not be randomly altered."
            )

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean dataset: standardize labels, remove duplicates, handle missing values."""
    original_len = len(df)

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Standardize label encoding if label exists
    if "label" in df.columns:
        def standardize_val(v):
            if pd.isnull(v):
                return np.nan
            if v in VALID_NORMAL_VALUES:
                return 0
            if v in VALID_SPOOFED_VALUES:
                return 1
            return np.nan

        df["label"] = df["label"].apply(standardize_val)
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)

    # Fill missing numeric values with column median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != "label" and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # Replace infinite values with NaN then fill with median
    df = df.replace([np.inf, -np.inf], np.nan)
    numeric_cols2 = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols2:
        if col != "label" and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    removed = original_len - len(df)
    if removed > 0:
        print(f"  Removed {removed} invalid/duplicate rows.")

    return df.reset_index(drop=True)


def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    sequence_length: int = SEQUENCE_LENGTH,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert flat feature matrix into overlapping temporal sequences.

    For sequence length L:
      X_seq[i] = X[i : i+L]          shape: (L, n_features)
      y_seq[i] = majority label in the sequence window

    This is appropriate for Bi-GRU which processes temporal sequences.
    """
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length + 1):
        X_seq.append(X[i : i + sequence_length])
        # Use the label of the last observation in the window
        y_seq.append(y[i + sequence_length - 1])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


def preprocess_dataset(
    filepath: str,
    sequence_length: int = SEQUENCE_LENGTH,
    test_size: float = None,
    val_size: float = None,
    fit_scaler: bool = True,
) -> Dict[str, Any]:
    """
    Full preprocessing pipeline. Returns a dictionary with:
      - X_train, X_val, X_test (sequences)
      - y_train, y_val, y_test
      - scaler (fitted StandardScaler)
      - feature_columns (list of feature names)
      - n_features (int)
      - class_distribution (dict)
    """
    if test_size is None:
        test_size = BIGRU_CONFIG["test_split"]
    if val_size is None:
        val_size = BIGRU_CONFIG["validation_split"]

    print("Step 1: Loading and validating dataset...")
    df = load_and_validate(filepath)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns.")

    print("Step 2: Cleaning data...")
    df = clean_data(df)
    print(f"  After cleaning: {len(df)} rows.")

    print("Step 3: Sorting by timestamp...")
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    print("Step 4: Engineering features...")
    df = engineer_features(df)

    print("Step 5: Extracting feature columns...")
    feature_columns = get_feature_columns(df, exclude_cols=["label", "timestamp", "id"])
    print(f"  Using {len(feature_columns)} features: {feature_columns[:5]}...")

    X = df[feature_columns].values.astype(np.float32)
    y = df["label"].values.astype(np.float32)

    print("Step 6: Train/val/test split (no data leakage)...")
    # First split off test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    # Then split remaining into train/val
    relative_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=relative_val_size, random_state=42, stratify=y_temp
    )

    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    print("Step 7: Normalizing features (scaler fitted on train only)...")
    if fit_scaler:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
    else:
        scaler = joblib.load(SCALER_PATH)
        X_train = scaler.transform(X_train)

    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    print("Step 8: Creating sequences...")
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
    X_val_seq, y_val_seq = create_sequences(X_val, y_val, sequence_length)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length)

    print(f"  Train sequences: {len(X_train_seq)}")
    print(f"  Val sequences:   {len(X_val_seq)}")
    print(f"  Test sequences:  {len(X_test_seq)}")

    class_distribution = {
        "normal": int((y == 0).sum()),
        "spoofed": int((y == 1).sum()),
        "total": len(y),
    }

    return {
        "X_train": X_train_seq,
        "X_val": X_val_seq,
        "X_test": X_test_seq,
        "y_train": y_train_seq,
        "y_val": y_val_seq,
        "y_test": y_test_seq,
        "scaler": scaler,
        "feature_columns": feature_columns,
        "n_features": len(feature_columns),
        "sequence_length": sequence_length,
        "class_distribution": class_distribution,
    }


def preprocess_for_prediction(
    filepath: str,
    sequence_length: int = SEQUENCE_LENGTH,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Preprocess a new CSV file for prediction using the saved scaler.
    Returns (X_sequences, y_true (if label exists), feature_columns)
    """
    print("Loading prediction dataset...")
    df = load_and_validate(filepath, require_label=False)
    df = clean_data(df)

    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    df = engineer_features(df)

    # Load saved feature config
    if FEATURE_CONFIG_PATH.exists():
        with open(FEATURE_CONFIG_PATH) as f:
            feature_config = json.load(f)
        feature_columns = feature_config["feature_columns"]
        # Only use features that exist in both saved config and uploaded CSV
        available = [c for c in feature_columns if c in df.columns]
        # Fill missing features with 0
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0.0
        feature_columns_to_use = feature_columns
    else:
        feature_columns_to_use = get_feature_columns(df, exclude_cols=["label", "timestamp", "id"])

    X = df[feature_columns_to_use].values.astype(np.float32)
    y = df["label"].values.astype(np.float32) if "label" in df.columns else None

    # Load saved scaler
    scaler = joblib.load(SCALER_PATH)
    X_scaled = scaler.transform(X)

    X_seq, y_seq = create_sequences(X_scaled, y if y is not None else np.zeros(len(X)), sequence_length)

    return X_seq, y_seq, feature_columns_to_use
