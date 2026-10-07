"""
Application Configuration
All hyperparameters and paths are defined here for easy modification.
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
TRAINED_MODELS_DIR = BASE_DIR / "trained_models"
DATASETS_DIR = BASE_DIR / "datasets"
UPLOADS_DIR = BASE_DIR / "uploads"

# Ensure directories exist
TRAINED_MODELS_DIR.mkdir(exist_ok=True)
DATASETS_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

# Database
DATABASE_URL = f"sqlite:///{BASE_DIR / 'gnss_detection.db'}"

# JWT Auth
SECRET_KEY = "gnss-spoofing-detection-secret-key-mca-project-2024"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# File upload
MAX_UPLOAD_SIZE_MB = 50
ALLOWED_EXTENSIONS = {".csv"}

# Sequence generation
SEQUENCE_LENGTH = 20

# Bi-GRU Model hyperparameters
BIGRU_CONFIG = {
    "sequence_length": SEQUENCE_LENGTH,
    "gru_units_1": 64,
    "gru_units_2": 32,
    "dropout_rate": 0.2,
    "dense_units": 32,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 25,
    "validation_split": 0.2,
    "test_split": 0.15,
    "early_stopping_patience": 4,
    "reduce_lr_patience": 5,
    "reduce_lr_factor": 0.5,
    "min_lr": 1e-6,
}

# Required GNSS feature columns (minimum required for processing)
REQUIRED_COLUMNS = ["label"]

# All possible GNSS feature columns
GNSS_FEATURE_COLUMNS = [
    "pseudorange",
    "carrier_phase",
    "cn0",           # C/N0 signal strength
    "doppler",
    "elevation",
    "azimuth",
    "latitude",
    "longitude",
    "altitude",
    "velocity",
]

# Model file paths
MODEL_PATH = TRAINED_MODELS_DIR / "bigru_model.pt"
SCALER_PATH = TRAINED_MODELS_DIR / "scaler.pkl"
FEATURE_CONFIG_PATH = TRAINED_MODELS_DIR / "feature_config.json"
MODEL_METADATA_PATH = TRAINED_MODELS_DIR / "model_metadata.json"

# Training status file
TRAINING_STATUS_PATH = TRAINED_MODELS_DIR / "training_status.json"

# CORS
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
]
