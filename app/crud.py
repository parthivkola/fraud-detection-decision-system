from __future__ import annotations

from typing import List, Optional

from sqlalchemy.orm import Session

from app.logger import logger
from app.models import PredictionBatch, PredictionResult


def create_prediction_batch(
    db: Session,
    total_transactions: int,
    flagged_fraud: int,
    threshold_used: float,
    model_version_id: Optional[int] = None,
) -> PredictionBatch:
    """Insert a new prediction batch and return it with generated fields populated."""
    logger.info(
        f"Creating prediction batch: total_transactions={total_transactions}, "
        f"flagged_fraud={flagged_fraud}, threshold={threshold_used}, "
        f"model_version_id={model_version_id}"
    )
    batch = PredictionBatch(
        total_transactions=total_transactions,
        flagged_fraud=flagged_fraud,
        threshold_used=threshold_used,
        model_version_id=model_version_id,
    )
    db.add(batch)
    db.commit()
    db.refresh(batch)
    logger.info(f"Prediction batch created with ID: {batch.id}")
    return batch


def create_prediction_results(
    db: Session,
    batch_id: int,
    predictions: List[dict],
) -> List[PredictionResult]:
    """Bulk-insert prediction results for a batch."""
    logger.info(f"Bulk inserting {len(predictions)} prediction results for batch {batch_id}")
    result_objects = [
        PredictionResult(
            batch_id=batch_id,
            row_index=p["row_index"],
            fraud_probability=p["fraud_probability"],
            is_fraud=p["is_fraud"],
            risk_level=p["risk_level"],
            decision=p["decision"],
        )
        for p in predictions
    ]

    db.add_all(result_objects)
    db.commit()

    for obj in result_objects:
        db.refresh(obj)

    logger.info(f"Inserted {len(result_objects)} prediction results for batch {batch_id}")
    return result_objects


def get_prediction_batch(db: Session, batch_id: int) -> Optional[PredictionBatch]:
    """Get a single batch by ID, or None if not found."""
    batch = (
        db.query(PredictionBatch)
        .filter(PredictionBatch.id == batch_id)
        .first()
    )
    if batch:
        logger.debug(f"Retrieved prediction batch {batch_id}")
    else:
        logger.warning(f"Prediction batch {batch_id} not found")
    return batch


def get_prediction_batches(
    db: Session,
    skip: int = 0,
    limit: int = 20,
) -> List[PredictionBatch]:
    """List recent prediction batches, newest first (paginated)."""
    batches = (
        db.query(PredictionBatch)
        .order_by(PredictionBatch.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    logger.debug(f"Retrieved {len(batches)} prediction batches (skip={skip}, limit={limit})")
    return batches


def get_results_for_batch(
    db: Session,
    batch_id: int,
) -> List[PredictionResult]:
    """Get all prediction results for a batch, ordered by row_index."""
    results = (
        db.query(PredictionResult)
        .filter(PredictionResult.batch_id == batch_id)
        .order_by(PredictionResult.row_index)
        .all()
    )
    logger.debug(f"Retrieved {len(results)} prediction results for batch {batch_id}")
    return results
