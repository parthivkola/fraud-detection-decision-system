"""
CRUD (Create, Read, Update, Delete) operations for predictions.

This layer sits between the router and the database.  The router
handles HTTP concerns (parsing requests, returning responses);
this module handles database concerns (inserting rows, querying).

Why separate from the router?
    - Keeps routes thin and focused on HTTP logic
    - CRUD functions can be reused by multiple routes
    - Easier to test — you can test DB logic without starting the API

All functions take a ``db: Session`` parameter.  The session is created
by the ``get_db()`` dependency in database.py and passed in by FastAPI.
"""

from __future__ import annotations

from typing import List, Optional

from sqlalchemy.orm import Session

from app.models import PredictionBatch, PredictionResult


# ---------------------------------------------------------------------------
# CREATE
# ---------------------------------------------------------------------------

def create_prediction_batch(
    db: Session,
    total_transactions: int,
    flagged_fraud: int,
    threshold_used: float,
) -> PredictionBatch:
    """
    Insert a new prediction batch row and return it.

    Steps:
      1. Create an ORM object (Python object, not yet in DB)
      2. db.add()     → stage it for insertion
      3. db.commit()  → actually write it to PostgreSQL
      4. db.refresh() → reload the object from DB so that auto-generated
                         fields (like ``id`` and ``created_at``) are populated
    """
    batch = PredictionBatch(
        total_transactions=total_transactions,
        flagged_fraud=flagged_fraud,
        threshold_used=threshold_used,
    )
    db.add(batch)
    db.commit()
    db.refresh(batch)
    return batch


def create_prediction_results(
    db: Session,
    batch_id: int,
    predictions: List[dict],
) -> List[PredictionResult]:
    """
    Bulk-insert all prediction results for a batch.

    ``predictions`` is a list of dicts, each with keys:
        row_index, fraud_probability, is_fraud, risk_level, decision

    We use db.add_all() to stage them all at once, then one db.commit()
    to write them in a single transaction.  This is much faster than
    committing each row individually.
    """
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

    # Refresh each object so their id/created_at fields are populated
    for obj in result_objects:
        db.refresh(obj)

    return result_objects


# ---------------------------------------------------------------------------
# READ
# ---------------------------------------------------------------------------

def get_prediction_batch(db: Session, batch_id: int) -> Optional[PredictionBatch]:
    """
    Get a single batch by its ID, or None if it doesn't exist.

    db.query(Model).filter(...) builds a SQL query like:
        SELECT * FROM prediction_batches WHERE id = :batch_id
    .first() returns the first result, or None if no rows match.
    """
    return (
        db.query(PredictionBatch)
        .filter(PredictionBatch.id == batch_id)
        .first()
    )


def get_prediction_batches(
    db: Session,
    skip: int = 0,
    limit: int = 20,
) -> List[PredictionBatch]:
    """
    List recent prediction batches, newest first.

    Pagination:
      - skip  = how many rows to skip (for page 2, skip=20)
      - limit = how many rows to return (page size)

    .order_by(PredictionBatch.created_at.desc()) sorts newest first.
    .offset(skip) skips the first N rows.
    .limit(limit) caps the result count.
    """
    return (
        db.query(PredictionBatch)
        .order_by(PredictionBatch.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )


def get_results_for_batch(
    db: Session,
    batch_id: int,
) -> List[PredictionResult]:
    """
    Get all individual prediction results for a given batch.

    Ordered by row_index so results come back in the same order
    as the original CSV rows.
    """
    return (
        db.query(PredictionResult)
        .filter(PredictionResult.batch_id == batch_id)
        .order_by(PredictionResult.row_index)
        .all()
    )
