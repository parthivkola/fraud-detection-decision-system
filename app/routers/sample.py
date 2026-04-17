"""Endpoint to serve random sample CSV from real dataset."""
from __future__ import annotations

import io
import random

import pandas as pd
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/api/v1", tags=["sample"])

SAMPLE_POOL = "data/sample_pool.csv"


@router.get("/sample-csv", summary="Download a random sample CSV from real data")
def download_sample_csv():
    """Return a random subset of 10-15 real transactions (mix of fraud + legit)."""
    pool = pd.read_csv(SAMPLE_POOL)

    fraud = pool[pool["Class"] == 1]
    legit = pool[pool["Class"] == 0]

    # Pick 3-5 fraud rows + 7-10 legit rows
    n_fraud = random.randint(3, min(5, len(fraud)))
    n_legit = random.randint(7, min(10, len(legit)))

    sample = pd.concat([
        fraud.sample(n_fraud),
        legit.sample(n_legit),
    ]).sample(frac=1)  # shuffle

    # Drop the Class column — user uploads V1-V28 + Amount only
    sample = sample.drop(columns=["Class"])

    buf = io.StringIO()
    sample.to_csv(buf, index=False)
    buf.seek(0)

    return StreamingResponse(
        io.BytesIO(buf.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=sample_transactions.csv"},
    )
