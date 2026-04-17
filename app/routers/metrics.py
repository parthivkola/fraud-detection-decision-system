from __future__ import annotations

import json

from fastapi import APIRouter, Depends, Request
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.auth import require_role
from app.config import settings
from app.database import get_db
from app.models import ModelVersion, PredictionBatch, PredictionResult, User
from app.schemas import MetricsResponse

router = APIRouter(prefix="/api/v1", tags=["metrics"])


def _load_model_metrics() -> dict:
    """Load precision, recall, F1, ROC-AUC from model metadata file."""
    defaults = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "roc_auc": 0.0}
    try:
        with open(settings.METADATA_PATH) as f:
            meta = json.load(f)
        test = meta.get("evaluation", meta.get("test_metrics", {}))
        return {
            "precision": round(test.get("precision", 0.0), 4),
            "recall": round(test.get("recall", 0.0), 4),
            "f1": round(test.get("f1", 0.0), 4),
            "roc_auc": round(test.get("roc_auc", 0.0), 4),
        }
    except Exception:
        return defaults


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Get system metrics (admin only)",
)
def get_metrics(
    request: Request,
    db: Session = Depends(get_db),
    _admin: User = Depends(require_role("admin")),
):
    """Return aggregate system metrics. Requires admin role."""
    import time

    # Counts
    total_predictions = db.query(func.count(PredictionResult.id)).scalar() or 0
    total_batches = db.query(func.count(PredictionBatch.id)).scalar() or 0

    flagged_fraud = (
        db.query(func.count(PredictionResult.id))
        .filter(PredictionResult.is_fraud.is_(True))
        .scalar()
        or 0
    )
    flagged_legitimate = total_predictions - flagged_fraud
    fraud_flag_rate = flagged_fraud / total_predictions if total_predictions > 0 else 0.0

    # Risk distribution
    risk_rows = (
        db.query(PredictionResult.risk_level, func.count(PredictionResult.id))
        .group_by(PredictionResult.risk_level)
        .all()
    )
    risk_distribution = {level: count for level, count in risk_rows}

    # Active model versions
    active_versions = (
        db.query(ModelVersion.version_tag)
        .filter(ModelVersion.is_active.is_(True))
        .all()
    )
    active_tags = [row[0] for row in active_versions]

    # Model quality metrics from saved metadata
    model_metrics = _load_model_metrics()

    # Uptime
    startup_time = getattr(request.app.state, "startup_time", time.time())
    uptime = time.time() - startup_time

    # Threshold
    threshold = getattr(request.app.state, "threshold", 0.5)

    return MetricsResponse(
        total_predictions=total_predictions,
        total_batches=total_batches,
        uptime_seconds=round(uptime, 2),
        active_model_versions=active_tags,
        threshold=round(threshold, 4),
        flagged_fraud=flagged_fraud,
        flagged_legitimate=flagged_legitimate,
        fraud_flag_rate=round(fraud_flag_rate, 4),
        model_precision=model_metrics["precision"],
        model_recall=model_metrics["recall"],
        model_f1=model_metrics["f1"],
        model_roc_auc=model_metrics["roc_auc"],
        risk_distribution=risk_distribution,
    )
