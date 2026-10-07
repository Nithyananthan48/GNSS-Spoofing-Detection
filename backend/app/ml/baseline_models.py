"""
Baseline Machine Learning Models for Comparison
================================================
Implements Logistic Regression, Random Forest, and SVM
evaluated on the SAME test set as the Bi-GRU model.

Features are flattened from sequences for traditional ML models.
"""
import numpy as np
from typing import Dict, Any
from app.ml.sklearn_compat import (
    LogisticRegression,
    RandomForestClassifier,
    SVC,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def flatten_sequences(X: np.ndarray) -> np.ndarray:
    """
    Flatten 3D sequence array (n_samples, seq_len, n_features)
    to 2D (n_samples, seq_len * n_features) for traditional ML models.
    """
    return X.reshape(X.shape[0], -1)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> Dict[str, float]:
    """Compute standard classification metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            metrics["roc_auc"] = 0.0
    else:
        metrics["roc_auc"] = 0.0
    return metrics


def train_baseline_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    """
    Train Logistic Regression, Random Forest, and SVM on the same splits
    and return their evaluation metrics.

    Args:
        X_train: Training sequences (3D)
        y_train: Training labels
        X_test: Test sequences (3D)
        y_test: Test labels

    Returns:
        Dict with model names as keys and metrics as values
    """
    # Flatten sequences for traditional ML
    X_train_flat = flatten_sequences(X_train)
    X_test_flat = flatten_sequences(X_test)

    results = {}

    # --- Logistic Regression ---
    print("  Training Logistic Regression...")
    try:
        lr = LogisticRegression(max_iter=300, random_state=42)
        lr.fit(X_train_flat, y_train)
        y_pred_lr = lr.predict(X_test_flat)
        y_proba_lr = lr.predict_proba(X_test_flat)[:, 1]
        results["Logistic Regression"] = compute_metrics(y_test, y_pred_lr, y_proba_lr)
        print(f"    LR Accuracy: {results['Logistic Regression']['accuracy']:.4f}")
    except Exception as e:
        print(f"    LR failed: {e}")
        results["Logistic Regression"] = {"error": str(e)}

    # --- Random Forest ---
    print("  Training Random Forest...")
    try:
        rf = RandomForestClassifier(n_estimators=25, max_depth=6, random_state=42)
        rf.fit(X_train_flat, y_train)
        y_pred_rf = rf.predict(X_test_flat)
        y_proba_rf = rf.predict_proba(X_test_flat)[:, 1]
        results["Random Forest"] = compute_metrics(y_test, y_pred_rf, y_proba_rf)
        print(f"    RF Accuracy: {results['Random Forest']['accuracy']:.4f}")
    except Exception as e:
        print(f"    RF failed: {e}")
        results["Random Forest"] = {"error": str(e)}

    # --- SVM ---
    print("  Training SVM...")
    try:
        max_svm_samples = min(2000, len(X_train_flat))
        idx = np.random.choice(len(X_train_flat), max_svm_samples, replace=False)
        svm = SVC(kernel="rbf", max_iter=250, random_state=42)
        svm.fit(X_train_flat[idx], y_train[idx])
        y_pred_svm = svm.predict(X_test_flat)
        y_proba_svm = svm.predict_proba(X_test_flat)[:, 1]
        results["SVM"] = compute_metrics(y_test, y_pred_svm, y_proba_svm)
        print(f"    SVM Accuracy: {results['SVM']['accuracy']:.4f}")
    except Exception as e:
        print(f"    SVM failed: {e}")
        results["SVM"] = {"error": str(e)}

    return results
