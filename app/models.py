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
)
from sqlalchemy.orm import relationship

from app.database import Base


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
