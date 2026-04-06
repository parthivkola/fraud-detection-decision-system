"""
Fraud prediction router.

POST /api/v1/fraud/predict
  - Accepts a CSV file upload
  - Preprocesses using the saved scaler (log1p + StandardScaler on Amount)
  - Runs XGBoost model inference
  - Returns per-row predictions with risk levels
  - Saves everything to the database for audit trail

GET /api/v1/fraud/history
  - List recent prediction batches (paginated)

GET /api/v1/fraud/history/{batch_id}
  - Get a specific batch with all its prediction results
"""

from __future__ import annotations

import io
from collections import Counter
from typing import List

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from sqlalchemy.orm import Session

from app.crud import (
    create_prediction_batch,
    create_prediction_results,
    get_prediction_batch,
    get_prediction_batches,
)
from app.database import get_db
from app.logger import logger
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

# Expected feature columns (V1-V28 + Amount)
REQUIRED_FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]


@router.post("/predict", response_model=PredictionResponse)
async def predict_fraud(
    request: Request,
    file: UploadFile = File(..., description="CSV file with transaction data"),
    # db is injected by FastAPI using the get_db dependency.
    # It's a live database session — you use it to read/write rows.
    db: Session = Depends(get_db),
):
    """
    Upload a CSV of transactions and get fraud predictions.

    The CSV must contain columns: V1-V28, Amount.
    The 'Time' and 'Class' columns are dropped if present.

    Results are saved to the database. The response includes a
    ``batch_id`` you can use to retrieve results later via
    GET /api/v1/fraud/history/{batch_id}.
    """
    # --- 1. Read CSV ---
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
    prediction_dicts = []  # We'll also collect dicts for the DB insert

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

        # Same data as a dict for crud.create_prediction_results()
        prediction_dicts.append({
            "row_index": idx,
            "fraud_probability": round(prob_float, 6),
            "is_fraud": is_fraud,
            "risk_level": risk_level,
            "decision": decision,
        })

    # --- 7. Summary ---
    risk_counts = dict(Counter(p.risk_level for p in predictions))
    flagged = sum(1 for p in predictions if p.is_fraud)

    # --- 8. Save to database ---
    # First, create the batch (parent row)
    batch = create_prediction_batch(
        db=db,
        total_transactions=len(predictions),
        flagged_fraud=flagged,
        threshold_used=threshold,
    )
    # Then, insert all individual results linked to this batch
    create_prediction_results(
        db=db,
        batch_id=batch.id,
        predictions=prediction_dicts,
    )

    logger.info(
        f"Prediction complete: {len(predictions)} txns, "
        f"{flagged} flagged, batch_id={batch.id}, "
        f"risk distribution: {risk_counts}"
    )

    return PredictionResponse(
        batch_id=batch.id,
        predictions=predictions,
        summary=PredictionSummary(
            total_transactions=len(predictions),
            flagged_fraud=flagged,
            risk_distribution=risk_counts,
        ),
        threshold_used=threshold,
    )


# ---------------------------------------------------------------------------
# History endpoints — read past predictions from the database
# ---------------------------------------------------------------------------

@router.get(
    "/history",
    response_model=List[PredictionBatchSummaryOut],
    summary="List recent prediction batches",
)
def list_prediction_history(
    # Query parameters for pagination.
    # skip = how many batches to skip (for page 2, set skip=20)
    # limit = how many batches per page (max 100 to prevent huge responses)
    skip: int = Query(0, ge=0, description="Number of batches to skip"),
    limit: int = Query(20, ge=1, le=100, description="Max batches to return"),
    db: Session = Depends(get_db),
):
    """
    List recent prediction batches (newest first), without individual results.

    Use the ``batch_id`` from the response to fetch full details via
    GET /api/v1/fraud/history/{batch_id}.
    """
    batches = get_prediction_batches(db, skip=skip, limit=limit)
    return batches


@router.get(
    "/history/{batch_id}",
    response_model=PredictionBatchOut,
    summary="Get a specific prediction batch with all results",
)
def get_prediction_detail(
    batch_id: int,
    db: Session = Depends(get_db),
):
    """
    Retrieve a specific prediction batch and all its individual results.

    Returns 404 if the batch_id doesn't exist.
    """
    batch = get_prediction_batch(db, batch_id=batch_id)
    if batch is None:
        raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found.")
    return batch