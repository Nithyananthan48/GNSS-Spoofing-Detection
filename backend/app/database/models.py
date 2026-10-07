"""
SQLAlchemy ORM Models
"""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from app.database.db import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    email = Column(String(150), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    datasets = relationship("Dataset", back_populates="user")
    predictions = relationship("Prediction", back_populates="user")


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    filepath = Column(String(500), nullable=False)
    rows_count = Column(Integer, default=0)
    features_count = Column(Integer, default=0)
    columns_info = Column(Text, nullable=True)  # JSON string
    uploaded_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="datasets")
    training_runs = relationship("TrainingRun", back_populates="dataset")
    predictions = relationship("Prediction", back_populates="dataset")


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=False)
    model_name = Column(String(100), default="Bi-GRU")
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    roc_auc = Column(Float, nullable=True)
    false_positive_rate = Column(Float, nullable=True)
    false_negative_rate = Column(Float, nullable=True)
    confusion_matrix = Column(Text, nullable=True)   # JSON
    training_history = Column(Text, nullable=True)   # JSON (loss/acc per epoch)
    baseline_metrics = Column(Text, nullable=True)   # JSON (LR/RF/SVM comparison)
    epochs_trained = Column(Integer, nullable=True)
    training_time_seconds = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    dataset = relationship("Dataset", back_populates="training_runs")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=True)
    filename = Column(String(255), nullable=True)
    prediction = Column(String(20), nullable=False)   # NORMAL or SPOOFED
    confidence = Column(Float, nullable=False)
    spoofed_count = Column(Integer, default=0)
    normal_count = Column(Integer, default=0)
    total_count = Column(Integer, default=0)
    spoofing_percentage = Column(Float, default=0.0)
    result_details = Column(Text, nullable=True)  # JSON
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="predictions")
    dataset = relationship("Dataset", back_populates="predictions")
