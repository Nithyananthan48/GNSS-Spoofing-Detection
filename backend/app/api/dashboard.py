"""
Dashboard Summary API.
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database.db import get_db
from app.database.models import Dataset, Prediction, TrainingRun
from app.utils.security import get_current_user
from app.ml.trainer import load_model_metadata

router = APIRouter(prefix="/api/dashboard", tags=["Dashboard"])


@router.get("/summary")
def dashboard_summary(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Return aggregated statistics for the dashboard."""
    user_id = current_user["user_id"]

    # Dataset stats
    datasets = db.query(Dataset).filter(Dataset.user_id == user_id).all()
    total_samples = sum(d.rows_count for d in datasets)

    # Prediction stats
    predictions = db.query(Prediction).filter(Prediction.user_id == user_id).all()
    total_preds = len(predictions)
    total_spoofed = sum(p.spoofed_count for p in predictions)
    total_normal = sum(p.normal_count for p in predictions)

    # Model metrics
    metadata = load_model_metadata()
    model_trained = metadata is not None
    accuracy = metadata["bigru_metrics"].get("accuracy", 0) if model_trained else None

    # Recent predictions
    recent = (
        db.query(Prediction)
        .filter(Prediction.user_id == user_id)
        .order_by(Prediction.created_at.desc())
        .limit(5)
        .all()
    )

    return {
        "total_datasets": len(datasets),
        "total_samples_uploaded": total_samples,
        "total_predictions_run": total_preds,
        "total_spoofed_detected": total_spoofed,
        "total_normal_detected": total_normal,
        "model_trained": model_trained,
        "model_accuracy": round(accuracy * 100, 2) if accuracy else None,
        "recent_predictions": [
            {
                "id": p.id,
                "filename": p.filename,
                "prediction": p.prediction,
                "confidence": p.confidence,
                "spoofing_percentage": p.spoofing_percentage,
                "created_at": p.created_at.isoformat() if p.created_at else None,
            }
            for p in recent
        ],
    }
