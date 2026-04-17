from __future__ import annotations

import io
import random
from collections import Counter
from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from sqlalchemy.orm import Session

from app.auth import get_current_user
from app.crud import (
    create_prediction_batch,
    create_prediction_results,
    get_prediction_batch,
    get_prediction_batches,
)
from app.database import get_db
from app.logger import logger
from app.models import User
from app.risk import assess_risk
from app.schemas import (
    PredictionBatchOut,
    PredictionBatchSummaryOut,
    PredictionResponse,
    PredictionSummary,
    TransactionPrediction,
)
from ml.preprocessing.transform_features import transform_new_data

router = APIRouter(prefix="/api/v1/fraud", tags=["fraud"])

REQUIRED_FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]


def _select_model(request: Request):
    """Select a model for inference.

    If multiple active model versions have been loaded (A/B test), a weighted
    random selection picks one per request.  Falls back to the default
    single-model loaded at startup.
    """
    loaded_versions = getattr(request.app.state, "loaded_versions", None)
    if loaded_versions and len(loaded_versions) > 0:
        tags = list(loaded_versions.keys())
        weights = [loaded_versions[t]["ab_weight"] for t in tags]
        chosen_tag = random.choices(tags, weights=weights, k=1)[0]
        v = loaded_versions[chosen_tag]
        return v["model"], v["scaler"], v["threshold"], v["version_id"], chosen_tag

    # Fallback: single model loaded at startup
    return (
        request.app.state.model,
        request.app.state.scaler,
        request.app.state.threshold,
        None,
        None,
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict_fraud(
    request: Request,
    file: UploadFile = File(..., description="CSV file with transaction data"),
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """
    Upload a CSV of transactions and get fraud predictions.

    The CSV must contain columns V1-V28 and Amount.
    Results are saved to the database; use the returned batch_id
    to retrieve them later.
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        logger.error(f"Failed to parse CSV: {e}")
        raise HTTPException(status_code=400, detail=f"Could not parse CSV file: {e}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV file is empty.")

    logger.info(f"Received CSV with {len(df)} rows, columns: {list(df.columns)}")

    for col in ["Time", "Class"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    missing = [c for c in REQUIRED_FEATURES if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {missing}",
        )

    df = df[REQUIRED_FEATURES]

    model, scaler, threshold, version_id, version_tag = _select_model(request)

    try:
        df_transformed = transform_new_data(df, scaler)
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {e}")

    try:
        probabilities = model.predict_proba(df_transformed)[:, 1]
    except Exception as e:
        logger.error(f"Model prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    predictions: List[TransactionPrediction] = []
    prediction_dicts: List[dict] = []

    for idx, prob in enumerate(probabilities):
        prob_float = float(prob)
        is_fraud = prob_float >= threshold
        risk_level, decision = assess_risk(prob_float, threshold)

        predictions.append(
            TransactionPrediction(
                row_index=idx,
                fraud_probability=round(prob_float, 6),
                is_fraud=is_fraud,
                risk_level=risk_level,
                decision=decision,
            )
        )

        prediction_dicts.append({
            "row_index": idx,
            "fraud_probability": round(prob_float, 6),
            "is_fraud": is_fraud,
            "risk_level": risk_level,
            "decision": decision,
        })

    risk_counts = dict(Counter(p.risk_level for p in predictions))
    flagged = sum(1 for p in predictions if p.is_fraud)

    batch = create_prediction_batch(
        db=db,
        total_transactions=len(predictions),
        flagged_fraud=flagged,
        threshold_used=threshold,
        model_version_id=version_id,
    )
    create_prediction_results(
        db=db,
        batch_id=batch.id,
        predictions=prediction_dicts,
    )

    logger.info(
        f"Prediction complete: {len(predictions)} txns, "
        f"{flagged} flagged, batch_id={batch.id}, "
        f"model={version_tag or 'default'}, "
        f"risk distribution: {risk_counts}"
    )

    return PredictionResponse(
        batch_id=batch.id,
        model_version=version_tag,
        predictions=predictions,
        summary=PredictionSummary(
            total_transactions=len(predictions),
            flagged_fraud=flagged,
            risk_distribution=risk_counts,
        ),
        threshold_used=threshold,
    )


@router.get(
    "/history",
    response_model=List[PredictionBatchSummaryOut],
    summary="List recent prediction batches",
)
def list_prediction_history(
    skip: int = Query(0, ge=0, description="Number of batches to skip"),
    limit: int = Query(20, ge=1, le=100, description="Max batches to return"),
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """List recent prediction batches (newest first), without individual results."""
    return get_prediction_batches(db, skip=skip, limit=limit)


@router.get(
    "/history/{batch_id}",
    response_model=PredictionBatchOut,
    summary="Get a specific prediction batch with all results",
)
def get_prediction_detail(
    batch_id: int,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """Retrieve a prediction batch and all its results. Returns 404 if not found."""
    batch = get_prediction_batch(db, batch_id=batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found.")
    return batch