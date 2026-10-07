"""
Dataset upload and management API.
"""
import os
import json
import shutil
from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import pandas as pd
import numpy as np

from app.database.db import get_db
from app.database.models import Dataset
from app.utils.security import get_current_user
from app.utils.validators import validate_csv_file, validate_file_size, safe_filename
from app.config import UPLOADS_DIR, GNSS_FEATURE_COLUMNS, BASE_DIR
from app.ml.preprocessor import (
    VALID_NORMAL_VALUES,
    VALID_SPOOFED_VALUES,
    detect_target_column_candidates,
    COMMON_TARGET_COLUMNS,
)

router = APIRouter(prefix="/api/dataset", tags=["Dataset"])

SAMPLE_TEMPLATE_PATH = BASE_DIR / "datasets" / "gnss_sample_template.csv"


@router.get("/sample-template")
def download_sample_template():
    """Download a verified sample GNSS CSV template with 'label' (0=Normal, 1=Spoofed)."""
    if not SAMPLE_TEMPLATE_PATH.exists():
        raise HTTPException(status_code=404, detail="Sample template not found.")
    return FileResponse(
        path=str(SAMPLE_TEMPLATE_PATH),
        filename="gnss_sample_template.csv",
        media_type="text/csv",
    )


@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    target_column: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """
    Upload and validate a GNSS CSV dataset.

    Requirements:
      1. Dataset must have a 'label' column (or user can map another column like 'class', 'attack').
      2. Validates that 'label' contains binary values: 0 (Normal) and 1 (Spoofed).
      3. Returns detected columns and dataset shape.
      4. Never generates random labels for user data or silently alters labels.
    """
    validate_csv_file(file)
    content = await file.read()
    validate_file_size(content)

    # Save file to user upload directory
    filename = safe_filename(file.filename)
    user_dir = UPLOADS_DIR / str(current_user["user_id"])
    user_dir.mkdir(parents=True, exist_ok=True)
    filepath = user_dir / filename

    with open(filepath, "wb") as f:
        f.write(content)

    # Parse CSV
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        if filepath.exists():
            os.remove(filepath)
        raise HTTPException(status_code=400, detail=f"Could not parse CSV file: {str(e)}")

    if df.empty:
        if filepath.exists():
            os.remove(filepath)
        raise HTTPException(status_code=400, detail="Uploaded CSV dataset is empty (0 rows).")

    dataset_shape = [int(len(df)), int(len(df.columns))]
    columns_list = df.columns.tolist()

    # Handle Target Column Mapping
    actual_target = target_column.strip() if target_column and target_column.strip() else None

    if actual_target:
        if actual_target not in df.columns:
            if filepath.exists():
                os.remove(filepath)
            raise HTTPException(
                status_code=400,
                detail={
                    "message": f"Specified target column '{actual_target}' was not found in dataset.",
                    "columns": columns_list,
                    "shape": dataset_shape,
                },
            )
        # Map user-specified target column to 'label'
        df = df.rename(columns={actual_target: "label"})
    elif "label" not in df.columns:
        # Check if alternative target candidate columns exist
        candidates = detect_target_column_candidates(columns_list)
        if filepath.exists():
            os.remove(filepath)

        if candidates:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": "MISSING_LABEL_WITH_CANDIDATES",
                    "message": (
                        f"Dataset does not have a 'label' column. "
                        f"Detected potential target column(s): {candidates}. "
                        f"Please select which column represents the attack label to map it to 'label'."
                    ),
                    "detected_candidates": candidates,
                    "columns": columns_list,
                    "shape": dataset_shape,
                },
            )
        else:
            raise HTTPException(
                status_code=400,
                detail={
                    "error_code": "MISSING_LABEL",
                    "message": (
                        "Dataset must contain a 'label' column where 0 = Normal and 1 = Spoofed. "
                        "No 'label' or common target column was found."
                    ),
                    "columns": columns_list,
                    "shape": dataset_shape,
                },
            )

    # Validate that 'label' contains valid binary values
    if df["label"].isnull().all():
        if filepath.exists():
            os.remove(filepath)
        raise HTTPException(
            status_code=400,
            detail={
                "message": "The 'label' column contains only null or missing values.",
                "columns": columns_list,
                "shape": dataset_shape,
            },
        )

    unique_raw_labels = df["label"].dropna().unique().tolist()
    invalid_labels = [
        str(val) for val in unique_raw_labels
        if val not in VALID_NORMAL_VALUES and val not in VALID_SPOOFED_VALUES
    ]

    if invalid_labels:
        if filepath.exists():
            os.remove(filepath)
        raise HTTPException(
            status_code=400,
            detail={
                "error_code": "INVALID_LABEL_VALUES",
                "message": (
                    f"Invalid values found in 'label' column: {invalid_labels[:5]}. "
                    f"The label column must contain binary values only: "
                    f"0 (Normal) and 1 (Spoofed)."
                ),
                "invalid_values": invalid_labels[:10],
                "columns": columns_list,
                "shape": dataset_shape,
            },
        )

    # Standardize label values to 0 and 1 without modifying meaning
    def standardize_label(val):
        if pd.isnull(val):
            return np.nan
        if val in VALID_NORMAL_VALUES:
            return 0
        if val in VALID_SPOOFED_VALUES:
            return 1
        return np.nan

    df["label"] = df["label"].apply(standardize_label)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    # Check that at least some data remains
    if len(df) == 0:
        if filepath.exists():
            os.remove(filepath)
        raise HTTPException(
            status_code=400,
            detail="No valid observations remaining after label parsing.",
        )

    # Save standardized CSV file so training & preprocessing read clean data
    df.to_csv(filepath, index=False)

    # Gather dataset analysis metadata
    available_features = [c for c in GNSS_FEATURE_COLUMNS if c in df.columns]
    label_counts = {str(k): int(v) for k, v in df["label"].value_counts().to_dict().items()}

    columns_info = {
        "all_columns": df.columns.tolist(),
        "shape": [len(df), len(df.columns)],
        "available_gnss_features": available_features,
        "label_distribution": label_counts,
        "missing_values": {str(k): int(v) for k, v in df.isnull().sum().to_dict().items()},
        "sample_preview": df.head(5).fillna("").to_dict(orient="records"),
    }

    # Save dataset metadata to DB
    dataset = Dataset(
        user_id=current_user["user_id"],
        filename=filename,
        filepath=str(filepath),
        rows_count=len(df),
        features_count=len(df.columns),
        columns_info=json.dumps(columns_info),
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    return {
        "dataset_id": dataset.id,
        "filename": filename,
        "rows_count": len(df),
        "features_count": len(df.columns),
        "shape": [len(df), len(df.columns)],
        "columns": df.columns.tolist(),
        "available_gnss_features": available_features,
        "label_distribution": label_counts,
        "missing_values": columns_info["missing_values"],
        "sample_preview": columns_info["sample_preview"],
        "mapped_target_column": actual_target,
        "message": f"Dataset '{filename}' uploaded and validated successfully ({len(df):,} rows × {len(df.columns)} columns).",
    }


@router.get("/list")
def list_datasets(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """List all datasets uploaded by the current user."""
    datasets = (
        db.query(Dataset)
        .filter(Dataset.user_id == current_user["user_id"])
        .order_by(Dataset.uploaded_at.desc())
        .all()
    )
    return [
        {
            "id": d.id,
            "filename": d.filename,
            "rows_count": d.rows_count,
            "features_count": d.features_count,
            "uploaded_at": d.uploaded_at.isoformat() if d.uploaded_at else None,
        }
        for d in datasets
    ]


@router.get("/{dataset_id}")
def get_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Get details of a specific dataset."""
    dataset = db.query(Dataset).filter(
        Dataset.id == dataset_id,
        Dataset.user_id == current_user["user_id"],
    ).first()
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found.")

    columns_info = json.loads(dataset.columns_info) if dataset.columns_info else {}
    return {
        "id": dataset.id,
        "filename": dataset.filename,
        "rows_count": dataset.rows_count,
        "features_count": dataset.features_count,
        "columns_info": columns_info,
        "uploaded_at": dataset.uploaded_at.isoformat() if dataset.uploaded_at else None,
    }
