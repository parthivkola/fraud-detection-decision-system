from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class TransactionPrediction(BaseModel):
    """Prediction result for one transaction row."""

    row_index: int = Field(..., description="0-based row index in the uploaded CSV")
    fraud_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Model's predicted probability of fraud"
    )
    is_fraud: bool = Field(..., description="True if probability >= threshold")
    risk_level: str = Field(..., description="LOW / MEDIUM / HIGH / CRITICAL")
    decision: str = Field(..., description="approve / review / block")


class PredictionSummary(BaseModel):
    """Aggregate stats for the batch."""

    total_transactions: int
    flagged_fraud: int
    risk_distribution: Dict[str, int] = Field(
        ..., description="Count per risk level, e.g. {'LOW': 90, 'CRITICAL': 2}"
    )


class PredictionResponse(BaseModel):
    """Full response from POST /api/v1/fraud/predict."""

    batch_id: int = Field(
        ..., description="ID of the saved batch — use this to look up results later"
    )
    predictions: List[TransactionPrediction]
    summary: PredictionSummary
    threshold_used: float = Field(
        ..., description="Decision threshold loaded from model metadata"
    )


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str
    model_loaded: bool


class PredictionResultOut(BaseModel):
    """Single prediction result from the history endpoint."""

    id: int
    row_index: int
    fraud_probability: float
    is_fraud: bool
    risk_level: str
    decision: str
    created_at: datetime

    model_config = {"from_attributes": True}


class PredictionBatchOut(BaseModel):
    """A prediction batch with all its results."""

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
