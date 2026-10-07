"""
Model Training and Evaluation API endpoints.
"""
import json
import threading
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.database.db import get_db
from app.database.models import Dataset, TrainingRun
from app.utils.security import get_current_user
from app.ml.trainer import train_model, get_training_status, load_model_metadata
from app.config import MODEL_METADATA_PATH, TRAINING_STATUS_PATH

router = APIRouter(prefix="/api/model", tags=["Model"])

# Global training lock to prevent concurrent training
_training_lock = threading.Lock()
_is_training = False


class TrainRequest(BaseModel):
    dataset_id: int


@router.post("/train")
def start_training(
    payload: TrainRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Start Bi-GRU model training in the background."""
    global _is_training

    if _is_training:
        raise HTTPException(
            status_code=409,
            detail="Training is already in progress. Please wait for it to complete."
        )

    dataset = db.query(Dataset).filter(
        Dataset.id == payload.dataset_id,
        Dataset.user_id == current_user["user_id"],
    ).first()

    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    def run_training_task():
        global _is_training
        _is_training = True
        try:
            result = train_model(dataset.filepath, dataset_id=dataset.id)

            # Save training run to DB using a new session
            from app.database.db import SessionLocal
            db_task = SessionLocal()
            try:
                bigru = result.get("bigru_metrics", {})
                run = TrainingRun(
                    dataset_id=dataset.id,
                    model_name="Bi-GRU",
                    accuracy=bigru.get("accuracy"),
                    precision=bigru.get("precision"),
                    recall=bigru.get("recall"),
                    f1_score=bigru.get("f1_score"),
                    roc_auc=bigru.get("roc_auc"),
                    false_positive_rate=bigru.get("false_positive_rate"),
                    false_negative_rate=bigru.get("false_negative_rate"),
                    confusion_matrix=json.dumps(bigru.get("confusion_matrix", [])),
                    training_history=json.dumps(result.get("training_history", {})),
                    baseline_metrics=json.dumps(result.get("baseline_metrics", {})),
                    epochs_trained=result.get("epochs_trained"),
                    training_time_seconds=result.get("training_time_seconds"),
                )
                db_task.add(run)
                db_task.commit()
            finally:
                db_task.close()
        except Exception as e:
            print(f"Training task failed: {e}")
        finally:
            _is_training = False

    background_tasks.add_task(run_training_task)

    return {"message": "Training started in background.", "dataset_id": payload.dataset_id}


@router.get("/status")
def training_status(current_user: dict = Depends(get_current_user)):
    """Poll current training status."""
    return get_training_status()


@router.get("/metrics")
def get_metrics(current_user: dict = Depends(get_current_user)):
    """Return the latest model metrics."""
    metadata = load_model_metadata()
    if metadata is None:
        return {
            "trained": False,
            "message": "Model not trained – metrics unavailable.",
        }
    return {
        "trained": True,
        "model_name": metadata.get("model_name", "Bi-GRU"),
        "bigru_metrics": metadata.get("bigru_metrics", {}),
        "baseline_metrics": metadata.get("baseline_metrics", {}),
        "training_history": metadata.get("training_history", {}),
        "epochs_trained": metadata.get("epochs_trained"),
        "training_time_seconds": metadata.get("training_time_seconds"),
        "trained_at": metadata.get("trained_at"),
        "class_distribution": metadata.get("class_distribution", {}),
    }
