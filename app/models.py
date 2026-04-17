from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from app.database import Base


class User(Base):
    """Application user with role-based access (analyst / admin)."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(120), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    role = Column(String(20), nullable=False, default="analyst")  # analyst | admin
    is_active = Column(Boolean, nullable=False, default=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} username={self.username} role={self.role}>"


class ModelVersion(Base):
    """Registered ML model version. Supports A/B testing via ab_weight."""

    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    version_tag = Column(String(50), unique=True, nullable=False)    # e.g. "v1.0"
    description = Column(Text, nullable=True)
    file_path = Column(String(255), nullable=False)                  # path to .joblib
    scaler_path = Column(String(255), nullable=False)
    metadata_path = Column(String(255), nullable=False)
    is_active = Column(Boolean, nullable=False, default=False)
    ab_weight = Column(Float, nullable=False, default=1.0)           # 0.0 – 1.0
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    batches = relationship("PredictionBatch", back_populates="model_version")

    def __repr__(self) -> str:
        return (
            f"<ModelVersion id={self.id} tag={self.version_tag} "
            f"active={self.is_active} weight={self.ab_weight}>"
        )


class PredictionBatch(Base):
    """One row per prediction API call. Stores batch-level metadata."""

    __tablename__ = "prediction_batches"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    total_transactions = Column(Integer, nullable=False)
    flagged_fraud = Column(Integer, nullable=False)
    threshold_used = Column(Float, nullable=False)

    model_version_id = Column(
        Integer,
        ForeignKey("model_versions.id"),
        nullable=True,
        index=True,
    )

    model_version = relationship("ModelVersion", back_populates="batches")
    results = relationship("PredictionResult", back_populates="batch")

    def __repr__(self) -> str:
        return (
            f"<PredictionBatch id={self.id} "
            f"total={self.total_transactions} "
            f"flagged={self.flagged_fraud}>"
        )


class PredictionResult(Base):
    """One row per transaction in a prediction batch."""

    __tablename__ = "prediction_results"

    id = Column(Integer, primary_key=True, index=True)
    batch_id = Column(
        Integer,
        ForeignKey("prediction_batches.id"),
        nullable=False,
        index=True,
    )
    row_index = Column(Integer, nullable=False)
    fraud_probability = Column(Float, nullable=False)
    is_fraud = Column(Boolean, nullable=False)
    risk_level = Column(String(10), nullable=False)   # LOW / MEDIUM / HIGH / CRITICAL
    decision = Column(String(10), nullable=False)      # approve / review / block

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    batch = relationship("PredictionBatch", back_populates="results")

    def __repr__(self) -> str:
        return (
            f"<PredictionResult row={self.row_index} "
            f"prob={self.fraud_probability:.4f} "
            f"risk={self.risk_level}>"
        )
