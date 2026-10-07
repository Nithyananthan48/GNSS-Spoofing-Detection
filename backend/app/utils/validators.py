"""
Input and file validators.
"""
import os
from pathlib import Path
from fastapi import HTTPException, UploadFile
from app.config import MAX_UPLOAD_SIZE_MB, ALLOWED_EXTENSIONS


def validate_csv_file(file: UploadFile) -> None:
    """Validate uploaded file: extension and size."""
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{ext}'. Only CSV files are allowed."
        )


def validate_file_size(content: bytes) -> None:
    """Ensure file content doesn't exceed max size."""
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Maximum allowed size is {MAX_UPLOAD_SIZE_MB} MB."
        )


def safe_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal attacks."""
    return os.path.basename(filename).replace("..", "").replace("/", "").replace("\\", "")


def validate_email(email: str) -> bool:
    """Basic email format validation."""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None
