"""
Pydantic schemas for fraud prediction request/response.

These schemas define the API contract (what goes in and out of the API).
They are separate from the SQLAlchemy models in models.py:
  - schemas.py  = shape of the JSON the client sees
  - models.py   = shape of the rows in the database

Why separate?
  You might store extra columns in the DB that you don't expose to clients,
  or combine data from multiple tables into one API response.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Single-row prediction result
# ---------------------------------------------------------------------------

class TransactionPrediction(BaseModel):
    """Prediction result for one transaction row."""

    row_index: int = Field(..., description="0-based row index in the uploaded CSV")
    fraud_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Model's predicted probability of fraud"
    )
    is_fraud: bool = Field(..., description="True if probability >= threshold")
    risk_level: str = Field(
        ..., description="LOW / MEDIUM / HIGH / CRITICAL"
    )
    decision: str = Field(
        ..., description="approve / review / block"
    )


# ---------------------------------------------------------------------------
# Batch response wrapping all rows
# ---------------------------------------------------------------------------

class PredictionSummary(BaseModel):
    """Aggregate stats for the batch."""

    total_transactions: int
    flagged_fraud: int
    risk_distribution: Dict[str, int] = Field(
        ..., description="Count per risk level, e.g. {'LOW': 90, 'CRITICAL': 2}"
    )


class PredictionResponse(BaseModel):
    """Full response returned by POST /api/v1/fraud/predict."""

    batch_id: int = Field(
        ..., description="ID of the saved batch — use this to look up results later"
    )
    predictions: List[TransactionPrediction]
    summary: PredictionSummary
    threshold_used: float = Field(
        ..., description="Decision threshold loaded from model metadata"
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    model_loaded: bool


# ---------------------------------------------------------------------------
# History endpoint schemas  (read from DB)
# ---------------------------------------------------------------------------
# These use ``model_config = {"from_attributes": True}`` which tells
# Pydantic: "You can build this schema from a SQLAlchemy model object,
# not just from a dict."  Without this, Pydantic would reject ORM objects.

class PredictionResultOut(BaseModel):
    """Single prediction result as returned by the history endpoint."""

    id: int
    row_index: int
    fraud_probability: float
    is_fraud: bool
    risk_level: str
    decision: str
    created_at: datetime

    model_config = {"from_attributes": True}


class PredictionBatchOut(BaseModel):
    """A prediction batch with all its individual results."""

    id: int
    created_at: datetime
    total_transactions: int
    flagged_fraud: int
    threshold_used: float
    results: List[PredictionResultOut] = []

    model_config = {"from_attributes": True}


class PredictionBatchSummaryOut(BaseModel):
    """Batch metadata without individual results (for listing)."""

    id: int
    created_at: datetime
    total_transactions: int
    flagged_fraud: int
    threshold_used: float

    model_config = {"from_attributes": True}
