from __future__ import annotations

import json
import time

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.auth import require_role
from app.config import settings
from app.database import get_db
from app.models import ModelVersion, PredictionBatch, PredictionResult, User
from app.schemas import MetricsResponse

router = APIRouter(prefix="/api/v1/metrics", tags=["metrics"])


def _model_metrics() -> dict:
    defaults = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0, "roc_auc": 0.0}
    try:
        with open(settings.METADATA_PATH) as f:
            meta = json.load(f)
        ev = meta.get("evaluation", meta.get("test_metrics", {}))
        return {
            "precision": round(ev.get("precision", 0.0), 4),
            "recall": round(ev.get("recall", 0.0), 4),
            "f1": round(ev.get("f1", 0.0), 4),
            "accuracy": round(ev.get("accuracy", 0.0), 4),
            "roc_auc": round(ev.get("roc_auc", 0.0), 4),
        }
    except Exception:
        return defaults


@router.get("", response_model=MetricsResponse)
@router.get("/", response_model=MetricsResponse)
def metrics(
    request: Request,
    model: str | None = Query(None, description="Filter by model version tag"),
    db: Session = Depends(get_db),
    _: User = Depends(require_role("admin", "analyst")),
):
    pq = db.query(PredictionResult)
    bq = db.query(PredictionBatch)
    rq = db.query(PredictionResult.risk_level, func.count(PredictionResult.id))

    if model:
        pq = pq.join(PredictionBatch).join(ModelVersion).filter(ModelVersion.version_tag == model)
        bq = bq.join(ModelVersion).filter(ModelVersion.version_tag == model)
        rq = rq.join(PredictionBatch).join(ModelVersion).filter(ModelVersion.version_tag == model)

    total = pq.count()
    batches = bq.count()
    fraud = pq.filter(PredictionResult.is_fraud.is_(True)).count()
    risk_dist = {lvl: cnt for lvl, cnt in rq.group_by(PredictionResult.risk_level).all()}
    for lvl in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        if lvl not in risk_dist:
            risk_dist[lvl] = 0

    active_tags = [
        r[0] for r in db.query(ModelVersion.version_tag).filter(ModelVersion.is_active.is_(True)).all()
    ]

    mm = _model_metrics()
    uptime = time.time() - getattr(request.app.state, "startup_time", time.time())
    threshold = getattr(request.app.state, "threshold", 0.5)

    return MetricsResponse(
        total_predictions=total,
        total_batches=batches,
        flagged_fraud=fraud,
        flagged_legitimate=total - fraud,
        fraud_flag_rate=round(fraud / total, 4) if total else 0.0,
        active_model_versions=active_tags,
        model_precision=mm["precision"],
        model_recall=mm["recall"],
        model_f1=mm["f1"],
        model_accuracy=mm["accuracy"],
        model_roc_auc=mm["roc_auc"],
        threshold=round(threshold, 4),
        uptime_seconds=round(uptime, 2),
        risk_distribution=risk_dist,
    )
