from __future__ import annotations
from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, EmailStr


# ── Auth ──────────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str
    email: Optional[str] = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    sub: Optional[str] = None
    role: str = "analyst"


class UserOut(BaseModel):
    id: int
    username: str
    email: str
    role: str

    model_config = {"from_attributes": True}


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool = True


# ── Models ────────────────────────────────────────────────────────────────────

class ModelVersionCreate(BaseModel):
    version_tag: str
    description: Optional[str] = None
    file_path: str
    scaler_path: str
    metadata_path: str


class ModelVersionUpdate(BaseModel):
    description: Optional[str] = None
    ab_weight: Optional[float] = None
    is_active: Optional[bool] = None


class ModelVersionOut(BaseModel):
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


class UpdateWeightRequest(BaseModel):
    ab_weight: float


# ── Predict ───────────────────────────────────────────────────────────────────

class PredictionRow(BaseModel):
    row_index: int
    fraud_probability: float
    is_fraud: bool
    risk_level: str   # LOW / MEDIUM / HIGH / CRITICAL
    decision: str     # approve / review / block


class PredictSummary(BaseModel):
    total_transactions: int
    flagged_fraud: int
    flagged_legitimate: int
    fraud_rate: float
    threshold_used: float
    model_version: Optional[str] = None


class PredictResponse(BaseModel):
    batch_id: int
    summary: PredictSummary
    predictions: list[PredictionRow]


class BatchOut(BaseModel):
    id: int
    created_at: datetime
    total_transactions: int
    flagged_fraud: int
    threshold_used: float
    model_version: Optional[Any] = None

    model_config = {"from_attributes": True}


# ── Metrics ───────────────────────────────────────────────────────────────────

class MetricsResponse(BaseModel):
    total_predictions: int
    total_batches: int
    flagged_fraud: int
    flagged_legitimate: int
    fraud_flag_rate: float
    active_model_versions: list[str]
    model_precision: float
    model_recall: float
    model_f1: float
    model_accuracy: float
    model_roc_auc: float
    threshold: float
    uptime_seconds: float
    risk_distribution: dict[str, int]
