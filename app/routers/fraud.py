"""
Fraud prediction router.

POST /api/v1/fraud/predict
  - Accepts a CSV file upload
  - Preprocesses using the saved scaler (log1p + StandardScaler on Amount)
  - Runs XGBoost model inference
  - Returns per-row predictions with risk levels
"""

from __future__ import annotations

import io
from collections import Counter

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.logger import logger
from app.risk import assess_risk
from app.schemas import (
    PredictionResponse,
    PredictionSummary,
    TransactionPrediction,
)
from ml.preprocessing.transform_features import transform_new_data

router = APIRouter(prefix="/api/v1/fraud", tags=["fraud"])

# Expected feature columns (V1–V28 + Amount)
REQUIRED_FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]


@router.post("/predict", response_model=PredictionResponse)
async def predict_fraud(
    request: Request,
    file: UploadFile = File(..., description="CSV file with transaction data"),
    # Future: current_user: User = Depends(get_current_user)
):
    """
    Upload a CSV of transactions and get fraud predictions.

    The CSV must contain columns: V1–V28, Amount.
    The 'Time' and 'Class' columns are dropped if present.
    """
    # --- 1. Read CSV ---
    if not file.filename.endswith(".csv"):
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

    # --- 2. Drop unnecessary columns ---
    for col in ["Time", "Class"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # --- 3. Validate required columns ---
    missing = [c for c in REQUIRED_FEATURES if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {missing}",
        )

    # Keep only required features in correct order
    df = df[REQUIRED_FEATURES]

    # --- 4. Preprocess (log1p + scale Amount) ---
    model = request.app.state.model
    scaler = request.app.state.scaler
    threshold = request.app.state.threshold

    try:
        df_transformed = transform_new_data(df, scaler)
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {e}")

    # --- 5. Predict ---
    try:
        probabilities = model.predict_proba(df_transformed)[:, 1]
    except Exception as e:
        logger.error(f"Model prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    # --- 6. Build per-row results ---
    predictions = []
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

    # --- 7. Summary ---
    risk_counts = dict(Counter(p.risk_level for p in predictions))
    flagged = sum(1 for p in predictions if p.is_fraud)

    logger.info(
        f"Prediction complete: {len(predictions)} txns, "
        f"{flagged} flagged, risk distribution: {risk_counts}"
    )

    return PredictionResponse(
        predictions=predictions,
        summary=PredictionSummary(
            total_transactions=len(predictions),
            flagged_fraud=flagged,
            risk_distribution=risk_counts,
        ),
        threshold_used=threshold,
    )