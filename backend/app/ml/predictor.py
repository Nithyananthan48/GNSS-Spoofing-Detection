"""
Prediction Pipeline
====================
Loads the saved PyTorch Bi-GRU model and scaler to run predictions on new GNSS data.
"""
import json
import numpy as np
import torch
from typing import Dict, Any, List

from app.config import (
    MODEL_PATH,
    SCALER_PATH,
    FEATURE_CONFIG_PATH,
    MODEL_METADATA_PATH,
    SEQUENCE_LENGTH,
)
from app.ml.preprocessor import preprocess_for_prediction
from app.ml.bigru_model import build_bigru_model


def load_bigru_model():
    """Load the saved Bi-GRU model from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "No trained model found. Please train the model first via the training page."
        )

    checkpoint = torch.load(str(MODEL_PATH), map_location=torch.device("cpu"), weights_only=False)
    seq_len = checkpoint.get("sequence_length", SEQUENCE_LENGTH)
    n_features = checkpoint["n_features"]
    cfg = checkpoint.get("config", None)

    model = build_bigru_model(sequence_length=seq_len, n_features=n_features)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, seq_len, n_features


def run_prediction(filepath: str) -> Dict[str, Any]:
    """
    Run GNSS spoofing detection on a new CSV file.

    Steps:
      1. Load saved model & scaler
      2. Preprocess the CSV into temporal sequences
      3. Run Bi-GRU inference
      4. Aggregate predictions & confidences
      5. Return structured detection report
    """
    # 1. Load model
    model, seq_len, n_features = load_bigru_model()

    # 2. Preprocess
    X_seq, y_true, feature_columns = preprocess_for_prediction(filepath, seq_len)

    if len(X_seq) == 0:
        raise ValueError(
            f"Insufficient data for sequence generation. "
            f"Dataset has fewer than {seq_len} rows required for temporal windowing."
        )

    # 3. Run inference
    with torch.no_grad():
        x_tensor = torch.tensor(X_seq, dtype=torch.float32)
        probs_tensor = model(x_tensor)
        y_proba = probs_tensor.cpu().numpy().flatten()

    y_pred = (y_proba >= 0.5).astype(int)

    # 4. Aggregate results
    total = len(y_pred)
    spoofed_count = int(y_pred.sum())
    normal_count = total - spoofed_count
    spoofing_pct = (spoofed_count / total * 100) if total > 0 else 0.0

    # Confidence calculation: how sure the model is about its decision for each window
    confidences = [
        float(p if pred == 1 else 1.0 - p) for p, pred in zip(y_proba, y_pred)
    ]
    avg_confidence = float(np.mean(confidences)) * 100 if confidences else 0.0

    # Overall classification: SPOOFED if >15% of sequence windows indicate attack
    overall_pred = "SPOOFED" if spoofing_pct > 15.0 else "NORMAL"

    # Per-sequence breakdown (capped for network response)
    sequence_details = [
        {
            "sequence_idx": int(i),
            "prediction": "SPOOFED" if pred == 1 else "NORMAL",
            "confidence": round(float(c) * 100, 2),
            "raw_probability": round(float(prob), 4),
        }
        for i, (pred, prob, c) in enumerate(zip(y_pred, y_proba, confidences))
    ]

    return {
        "overall_prediction": overall_pred,
        "confidence": round(avg_confidence, 2),
        "total_sequences": total,
        "spoofed_count": spoofed_count,
        "normal_count": normal_count,
        "spoofing_percentage": round(spoofing_pct, 2),
        "sequence_details": sequence_details[:500],
        "feature_columns_used": feature_columns,
        "model_info": {
            "name": "Bi-GRU (Bidirectional Gated Recurrent Unit)",
            "sequence_length": seq_len,
            "n_features": len(feature_columns),
        },
    }
