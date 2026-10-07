"""
Model Training Pipeline
========================
Orchestrates the full training workflow using PyTorch Bi-GRU:
  1. Preprocess dataset
  2. Build Bi-GRU model
  3. Train with EarlyStopping, ReduceLROnPlateau, Checkpointing
  4. Evaluate on test set
  5. Train baseline models for comparison
  6. Save model, scaler, and metadata
  7. Update training status
"""
import json
import time
import copy
import numpy as np
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from app.ml.sklearn_compat import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from app.config import (
    BIGRU_CONFIG,
    MODEL_PATH,
    SCALER_PATH,
    FEATURE_CONFIG_PATH,
    MODEL_METADATA_PATH,
    TRAINING_STATUS_PATH,
    TRAINED_MODELS_DIR,
)
from app.ml.preprocessor import preprocess_dataset
from app.ml.bigru_model import build_bigru_model, model_summary_str
from app.ml.baseline_models import train_baseline_models


def update_status(status: str, progress: int = 0, message: str = ""):
    """Write training status to a JSON file for polling."""
    data = {
        "status": status,
        "progress": progress,
        "message": message,
        "updated_at": datetime.utcnow().isoformat(),
    }
    with open(TRAINING_STATUS_PATH, "w") as f:
        json.dump(data, f)


def get_training_status() -> Dict[str, Any]:
    """Read current training status."""
    if not TRAINING_STATUS_PATH.exists():
        return {"status": "idle", "progress": 0, "message": "No training started yet."}
    with open(TRAINING_STATUS_PATH) as f:
        return json.load(f)


def train_model(filepath: str, dataset_id: int = None) -> Dict[str, Any]:
    """
    Full training pipeline.

    Args:
        filepath: Path to the training CSV file
        dataset_id: Optional dataset DB ID for logging

    Returns:
        Dictionary with all training results and metrics
    """
    start_time = time.time()
    update_status("preprocessing", 5, "Loading and preprocessing dataset...")

    try:
        # ------------------------------------------------------------------ #
        # Step 1: Preprocess
        # ------------------------------------------------------------------ #
        data = preprocess_dataset(filepath, fit_scaler=True)
        X_train = data["X_train"]
        X_val   = data["X_val"]
        X_test  = data["X_test"]
        y_train = data["y_train"]
        y_val   = data["y_val"]
        y_test  = data["y_test"]
        scaler  = data["scaler"]
        feature_columns = data["feature_columns"]
        n_features = data["n_features"]
        seq_len = data["sequence_length"]

        update_status("building_model", 15, f"Building Bi-GRU network ({seq_len} steps × {n_features} features)...")

        # ------------------------------------------------------------------ #
        # Step 2: Build model
        # ------------------------------------------------------------------ #
        model = build_bigru_model(seq_len, n_features)
        print(f"\n{model_summary_str(model)}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # PyTorch DataLoaders
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).unsqueeze(1),
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32).unsqueeze(1),
        )
        test_dataset = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32).unsqueeze(1),
        )

        batch_size = BIGRU_CONFIG["batch_size"]
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=BIGRU_CONFIG["learning_rate"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=BIGRU_CONFIG["reduce_lr_factor"],
            patience=BIGRU_CONFIG["reduce_lr_patience"],
            min_lr=BIGRU_CONFIG["min_lr"],
        )

        # ------------------------------------------------------------------ #
        # Step 3: Train loop with EarlyStopping & Checkpointing
        # ------------------------------------------------------------------ #
        update_status("training", 20, "Training Bi-GRU model...")
        epochs = BIGRU_CONFIG["epochs"]
        patience = BIGRU_CONFIG["early_stopping_patience"]

        best_val_loss = float("inf")
        best_model_weights = copy.deepcopy(model.state_dict())
        epochs_without_improvement = 0

        history = {
            "loss": [],
            "val_loss": [],
            "accuracy": [],
            "val_accuracy": [],
        }

        actual_epochs_trained = 0

        for epoch in range(1, epochs + 1):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * bx.size(0)
                preds = (out >= 0.5).float()
                train_correct += (preds == by).sum().item()
                train_total += bx.size(0)

            epoch_train_loss = train_loss / train_total
            epoch_train_acc = train_correct / train_total

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    out = model(bx)
                    loss = criterion(out, by)

                    val_loss += loss.item() * bx.size(0)
                    preds = (out >= 0.5).float()
                    val_correct += (preds == by).sum().item()
                    val_total += bx.size(0)

            epoch_val_loss = val_loss / val_total
            epoch_val_acc = val_correct / val_total

            scheduler.step(epoch_val_loss)

            history["loss"].append(float(epoch_train_loss))
            history["val_loss"].append(float(epoch_val_loss))
            history["accuracy"].append(float(epoch_train_acc))
            history["val_accuracy"].append(float(epoch_val_acc))

            actual_epochs_trained = epoch

            # Checkpoint & Early stopping
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_model_weights = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

            # Progress update
            progress_pct = int(20 + (epoch / epochs) * 50)
            if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
                print(f"Epoch {epoch:02d}/{epochs} - Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")
                update_status("training", progress_pct, f"Training epoch {epoch}/{epochs} (Loss: {epoch_train_loss:.4f}, Val Acc: {epoch_val_acc:.2%})")

        # Restore best weights
        model.load_state_dict(best_model_weights)
        update_status("evaluating", 75, f"Training complete ({actual_epochs_trained} epochs). Evaluating...")

        # ------------------------------------------------------------------ #
        # Step 4: Evaluate Bi-GRU on test set
        # ------------------------------------------------------------------ #
        model.eval()
        all_probs = []
        with torch.no_grad():
            for bx, by in test_loader:
                bx = bx.to(device)
                out = model(bx)
                all_probs.extend(out.cpu().numpy().flatten())

        y_proba = np.array(all_probs)
        y_pred = (y_proba >= 0.5).astype(int)
        y_true = y_test.astype(int)

        acc  = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec  = float(recall_score(y_true, y_pred, zero_division=0))
        f1   = float(f1_score(y_true, y_pred, zero_division=0))
        try:
            auc = float(roc_auc_score(y_true, y_proba))
        except Exception:
            auc = 0.0

        cm = confusion_matrix(y_true, y_pred).tolist()
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel() if len(np.unique(y_true)) == 2 else (0, 0, 0, 0)
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

        bigru_metrics = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": auc,
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
            "confusion_matrix": cm,
        }

        # ------------------------------------------------------------------ #
        # Step 5: Baseline models comparison
        # ------------------------------------------------------------------ #
        update_status("baseline_models", 80, "Training baseline ML models (LR, RF, SVM) for comparison...")
        baseline_metrics = train_baseline_models(X_train, y_train, X_test, y_test)

        # ------------------------------------------------------------------ #
        # Step 6: Save model, scaler & metadata
        # ------------------------------------------------------------------ #
        update_status("saving", 95, "Saving model and artifacts...")

        # Save PyTorch model state and architecture config
        torch.save({
            "sequence_length": seq_len,
            "n_features": n_features,
            "state_dict": model.state_dict(),
            "config": BIGRU_CONFIG,
        }, str(MODEL_PATH))
        print(f"  Bi-GRU Model saved to: {MODEL_PATH}")

        # Save scaler
        joblib.dump(scaler, SCALER_PATH)
        print(f"  Scaler saved to: {SCALER_PATH}")

        # Save feature config
        feature_config = {
            "feature_columns": feature_columns,
            "n_features": n_features,
            "sequence_length": seq_len,
        }
        with open(FEATURE_CONFIG_PATH, "w") as f:
            json.dump(feature_config, f, indent=2)

        # Save model metadata
        training_time = time.time() - start_time
        metadata = {
            "model_name": "Bi-GRU",
            "dataset_id": dataset_id,
            "sequence_length": seq_len,
            "n_features": n_features,
            "feature_columns": feature_columns,
            "epochs_trained": actual_epochs_trained,
            "training_time_seconds": round(training_time, 2),
            "trained_at": datetime.utcnow().isoformat(),
            "bigru_metrics": bigru_metrics,
            "baseline_metrics": baseline_metrics,
            "training_history": history,
            "class_distribution": data["class_distribution"],
            "config": BIGRU_CONFIG,
        }
        with open(MODEL_METADATA_PATH, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n==========================================")
        print(f"Training successfully completed!")
        print(f"  Bi-GRU Accuracy:   {acc:.4f} ({acc*100:.2f}%)")
        print(f"  Bi-GRU F1 Score:   {f1:.4f}")
        print(f"  Bi-GRU ROC-AUC:    {auc:.4f}")
        print(f"  Training time:     {training_time:.1f}s")
        print(f"==========================================")

        update_status("completed", 100, f"Training complete! Accuracy: {acc:.2%}")
        return metadata

    except Exception as e:
        import traceback
        error_msg = f"Training failed: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        update_status("failed", 0, str(e))
        raise


def load_model_metadata() -> Dict[str, Any]:
    """Load and return saved model metadata."""
    if not MODEL_METADATA_PATH.exists():
        return None
    with open(MODEL_METADATA_PATH) as f:
        return json.load(f)
