"""
SQLAlchemy ORM models — define the database tables.

Each class here maps to one PostgreSQL table.  SQLAlchemy calls these
"models" (not to be confused with your ML model).  The class attributes
become table columns.

Relationship between the two tables:

    PredictionBatch  1 ──── *  PredictionResult
    (one API call)              (one per transaction row in the CSV)

Why two tables instead of one?
    Batch-level info (threshold, total count, timestamp) would be
    duplicated on every row if we used a single table.  Splitting
    them keeps the data normalized and queries fast.
"""

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
    """
    One row per API call to POST /predict.

    Stores batch-level metadata so you can answer questions like:
      - "How many predictions did we make today?"
      - "What threshold were we using last week?"
    """

    # __tablename__ tells SQLAlchemy what to name the actual SQL table.
    __tablename__ = "prediction_batches"

    # --- Columns ---
    # Column(Type, ...) maps to a SQL column.
    # primary_key=True  → this is the table's primary key (auto-increments)
    # nullable=False    → NOT NULL constraint in the database
    id = Column(Integer, primary_key=True, index=True)

    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    total_transactions = Column(Integer, nullable=False)
    flagged_fraud = Column(Integer, nullable=False)
    threshold_used = Column(Float, nullable=False)

    # --- Relationship ---
    # This does NOT create a column in the DB.  It tells SQLAlchemy:
    #   "When I load a PredictionBatch, also give me easy access to
    #    all related PredictionResult rows via  batch.results"
    #
    # back_populates="batch" means the PredictionResult model has a
    # matching attribute called "batch" that points back here.
    results = relationship("PredictionResult", back_populates="batch")

    def __repr__(self) -> str:
        return (
            f"<PredictionBatch id={self.id} "
            f"total={self.total_transactions} "
            f"flagged={self.flagged_fraud}>"
        )


class PredictionResult(Base):
    """
    One row per transaction in a prediction batch.

    This is your audit trail — every single prediction the API has
    ever made is recorded here.
    """

    __tablename__ = "prediction_results"

    id = Column(Integer, primary_key=True, index=True)

    # --- Foreign key ---
    # ForeignKey("prediction_batches.id") creates a link to the
    # PredictionBatch table.  The database enforces this constraint:
    # you can't insert a result with a batch_id that doesn't exist.
    batch_id = Column(
        Integer,
        ForeignKey("prediction_batches.id"),
        nullable=False,
        index=True,  # index for fast lookups by batch
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

    # --- Relationship back to parent ---
    # back_populates="results" links this to PredictionBatch.results
    batch = relationship("PredictionBatch", back_populates="results")

    def __repr__(self) -> str:
        return (
            f"<PredictionResult row={self.row_index} "
            f"prob={self.fraud_probability:.4f} "
            f"risk={self.risk_level}>"
        )
