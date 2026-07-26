from __future__ import annotations

import io
import random

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.database import get_db
from app.models import ModelVersion, PredictionBatch, PredictionResult, User
from app.risk import decision as make_decision
from app.risk import risk_level as make_risk
from app.schemas import BatchOut, PredictResponse, PredictionRow
from ml.preprocessing.transform_features import transform_new_data

router = APIRouter(prefix="/api/predict", tags=["predict"])

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]


def _pick_model(state, override: str | None):
    """Return (model, scaler, threshold, tag) from app.state.models."""
    models: dict = getattr(state, "models", {})
    if not models:
        # Fallback to legacy single model
        return state.model, state.scaler, state.threshold, None

    if override:
        if override not in models:
            raise HTTPException(400, f"Model '{override}' not found or not loaded")
        v = models[override]
        return v["model"], v["scaler"], v["threshold"], override

    # Weighted selection from active models
    active = {tag: v for tag, v in models.items() if v.get("is_active", True)}
    if not active:
        raise HTTPException(503, "No active model versions available")
    tags = list(active.keys())
    weights = [active[t]["ab_weight"] for t in tags]
    chosen = random.choices(tags, weights=weights, k=1)[0]
    v = active[chosen]
    return v["model"], v["scaler"], v["threshold"], chosen


@router.post("", response_model=PredictResponse)
async def predict(
    request: Request,
    file: UploadFile = File(...),
    model: str | None = Query(None, description="Pin a specific model version tag"),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a .csv file")

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(400, f"Could not parse CSV: {e}")

    if df.empty:
        raise HTTPException(400, "CSV is empty")

    # Drop label columns if present
    df = df.drop(columns=[c for c in ["Time", "Class"] if c in df.columns])

    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise HTTPException(400, f"Missing columns: {missing}")

    df = df[FEATURES]
    clf, scaler, threshold, version_tag = _pick_model(request.app.state, model)

    try:
        df_t = transform_new_data(df, scaler)
        probs = clf.predict_proba(df_t)[:, 1]
    except Exception as e:
        raise HTTPException(500, f"Prediction error: {e}")

    rows: list[PredictionRow] = []
    for i, prob in enumerate(probs):
        rows.append(PredictionRow(
            row_index=i,
            fraud_probability=float(prob),
            is_fraud=prob >= threshold,
            risk_level=make_risk(float(prob), threshold),
            decision=make_decision(float(prob), threshold),
        ))

    flagged = sum(1 for r in rows if r.is_fraud)

    # Resolve version FK
    version_id = None
    if version_tag:
        mv = db.query(ModelVersion).filter(ModelVersion.version_tag == version_tag).first()
        version_id = mv.id if mv else None

    batch = PredictionBatch(
        total_transactions=len(rows),
        flagged_fraud=flagged,
        threshold_used=threshold,
        model_version_id=version_id,
    )
    db.add(batch)
    db.flush()

    db.bulk_save_objects([
        PredictionResult(
            batch_id=batch.id,
            row_index=r.row_index,
            fraud_probability=r.fraud_probability,
            is_fraud=r.is_fraud,
            risk_level=r.risk_level,
            decision=r.decision,
        )
        for r in rows
    ])
    db.commit()
    db.refresh(batch)

    return PredictResponse(
        batch_id=batch.id,
        model_version=version_tag,
        threshold_used=threshold,
        total=len(rows),
        flagged=flagged,
        predictions=rows,
    )


@router.get("/history", response_model=list[BatchOut])
def history(
    limit: int = Query(50, le=200),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
):
    batches = (
        db.query(PredictionBatch)
        .order_by(PredictionBatch.created_at.desc())
        .limit(limit)
        .all()
    )
    result = []
    for b in batches:
        tag = b.model_version.version_tag if b.model_version else None
        result.append(BatchOut(
            id=b.id,
            created_at=b.created_at,
            total_transactions=b.total_transactions,
            flagged_fraud=b.flagged_fraud,
            threshold_used=b.threshold_used,
            model_version=tag,
        ))
    return result


@router.get("/sample-csv")
def sample_csv():
    """Generate a synthetic sample CSV for testing."""
    n = random.randint(12, 18)
    data: dict = {}
    for i in range(1, 29):
        std = random.uniform(0.9, 1.4)
        col = np.round(np.random.normal(0.0, std, n), 6)
        # Inject outliers on high-signal features to simulate fraud
        if i in [1, 3, 4, 10, 14, 17]:
            for idx in random.sample(range(n), k=random.randint(2, 4)):
                col[idx] = round(random.choice([-1, 1]) * random.uniform(4, 12), 6)
        data[f"V{i}"] = col
    data["Amount"] = [
        round(random.uniform(5, 150), 2) if random.random() < 0.8
        else round(random.uniform(200, 1500), 2)
        for _ in range(n)
    ]
    buf = io.StringIO()
    pd.DataFrame(data).to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        io.BytesIO(buf.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=sample_transactions.csv"},
    )
