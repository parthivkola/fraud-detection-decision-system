"""Endpoint to generate and serve sample CSV transactions."""
from __future__ import annotations

import io
import random

import numpy as np
import pandas as pd
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/api/v1", tags=["sample"])


@router.get("/sample-csv", summary="Download a random sample CSV for testing")
def download_sample_csv():
    """Generate and return 12-16 random synthetic transactions with realistic values."""
    n_rows = random.randint(12, 16)
    data = {}
    
    # Generate V1-V28 features (PCA components centered around 0)
    for i in range(1, 29):
        std = random.uniform(0.8, 1.5)
        data[f"V{i}"] = np.round(np.random.normal(loc=0.0, scale=std, size=n_rows), 6)
        
        # Inject fraudulent outlier rows with extreme PCA values on predictive features
        if i in [1, 3, 4, 10, 14, 17]:
            outlier_indices = random.sample(range(n_rows), k=random.randint(2, 4))
            for idx in outlier_indices:
                data[f"V{i}"][idx] = round(random.choice([-1.0, 1.0]) * random.uniform(4.0, 12.0), 6)

    # Generate realistic transaction amounts ($5 - $1200)
    amounts = []
    for _ in range(n_rows):
        if random.random() < 0.8:
            amounts.append(round(random.uniform(5.0, 150.0), 2))
        else:
            amounts.append(round(random.uniform(200.0, 1200.0), 2))
    data["Amount"] = amounts

    sample = pd.DataFrame(data)

    buf = io.StringIO()
    sample.to_csv(buf, index=False)
    buf.seek(0)

    return StreamingResponse(
        io.BytesIO(buf.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=sample_transactions.csv"},
    )
