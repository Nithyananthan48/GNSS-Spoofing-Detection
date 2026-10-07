"""
Prediction API endpoints.
"""
import os
import json
from pathlib import Path
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from sqlalchemy.orm import Session

from app.database.db import get_db
from app.database.models import Prediction, Dataset
from app.utils.security import get_current_user
from app.utils.validators import validate_csv_file, validate_file_size, safe_filename
from app.config import UPLOADS_DIR
from app.ml.predictor import run_prediction

router = APIRouter(prefix="/api", tags=["Prediction"])


@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Upload a GNSS CSV and run spoofing detection."""
    validate_csv_file(file)
    content = await file.read()
    validate_file_size(content)

    filename = safe_filename(file.filename)
    user_dir = UPLOADS_DIR / str(current_user["user_id"]) / "predictions"
    user_dir.mkdir(parents=True, exist_ok=True)
    filepath = user_dir / filename

    with open(filepath, "wb") as f:
        f.write(content)

    try:
        result = run_prediction(str(filepath))
    except FileNotFoundError as e:
        os.remove(filepath)
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        os.remove(filepath)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        os.remove(filepath)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # Save to DB
    prediction_record = Prediction(
        user_id=current_user["user_id"],
        filename=filename,
        prediction=result["overall_prediction"],
        confidence=result["confidence"],
        spoofed_count=result["spoofed_count"],
        normal_count=result["normal_count"],
        total_count=result["total_sequences"],
        spoofing_percentage=result["spoofing_percentage"],
        result_details=json.dumps({
            k: v for k, v in result.items()
            if k != "sequence_details"  # don't store large array in DB
        }),
    )
    db.add(prediction_record)
    db.commit()
    db.refresh(prediction_record)

    result["prediction_id"] = prediction_record.id
    result["saved_at"] = prediction_record.created_at.isoformat()

    return result


@router.get("/predictions")
def get_predictions(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Get prediction history for the current user."""
    predictions = (
        db.query(Prediction)
        .filter(Prediction.user_id == current_user["user_id"])
        .order_by(Prediction.created_at.desc())
        .limit(100)
        .all()
    )
    return [
        {
            "id": p.id,
            "filename": p.filename,
            "prediction": p.prediction,
            "confidence": p.confidence,
            "spoofed_count": p.spoofed_count,
            "normal_count": p.normal_count,
            "total_count": p.total_count,
            "spoofing_percentage": p.spoofing_percentage,
            "created_at": p.created_at.isoformat() if p.created_at else None,
        }
        for p in predictions
    ]
