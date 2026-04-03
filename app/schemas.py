"""
Pydantic schemas for fraud prediction request/response.

These schemas define the API contract. They are independent of the
database models so auth/DB can be added later without changing the API shape.
"""

from __future__ import annotations

from typing import Dict, List

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
