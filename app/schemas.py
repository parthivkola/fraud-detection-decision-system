from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, EmailStr, Field


# ── Auth Schemas ──────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    """Payload for POST /auth/register."""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=128)


class UserOut(BaseModel):
    """Public user representation (never exposes the password hash)."""
    id: int
    username: str
    email: str
    role: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class TokenResponse(BaseModel):
    """Returned by POST /auth/login."""
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Decoded JWT payload."""
    sub: str          # username
    role: str


# ── Model Version Schemas ────────────────────────────────────────────────────

class ModelVersionCreate(BaseModel):
    """Payload for registering a new model version."""
    version_tag: str = Field(..., max_length=50, examples=["v2.0"])
    description: Optional[str] = None
    file_path: str
    scaler_path: str
    metadata_path: str


class ModelVersionUpdate(BaseModel):
    """Partial update for a model version."""
    description: Optional[str] = None
    ab_weight: Optional[float] = Field(None, ge=0.0, le=1.0)


class ModelVersionOut(BaseModel):
    """Public model version representation."""
    id: int
    version_tag: str
    description: Optional[str]
    file_path: str
    scaler_path: str
    metadata_path: str
    is_active: bool
    ab_weight: float
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Fraud / Prediction Schemas ───────────────────────────────────────────────

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
    model_version: Optional[str] = Field(
        None, description="Version tag of the model used for this prediction"
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
    model_version_id: Optional[int] = None
    results: List[PredictionResultOut] = []

    model_config = {"from_attributes": True}


class PredictionBatchSummaryOut(BaseModel):
    """Batch metadata without individual results (for listing)."""

    id: int
    created_at: datetime
    total_transactions: int
    flagged_fraud: int
    threshold_used: float
    model_version_id: Optional[int] = None

    model_config = {"from_attributes": True}


# ── Metrics Schema ───────────────────────────────────────────────────────────

class MetricsResponse(BaseModel):
    """Response from GET /api/v1/metrics."""
    # Operational
    total_predictions: int
    total_batches: int
    uptime_seconds: float
    active_model_versions: list[str]
    threshold: float

    # Detection stats
    flagged_fraud: int
    flagged_legitimate: int
    fraud_flag_rate: float              # flagged / total

    # Model quality (from saved metadata)
    model_precision: float
    model_recall: float
    model_f1: float
    model_roc_auc: float

    # Risk breakdown
    risk_distribution: dict[str, int]

