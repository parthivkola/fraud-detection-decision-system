from __future__ import annotations

import io
import random

import numpy as np
import pandas as pd
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/api/v1", tags=["sample"])


def _generate_synthetic_csv() -> StreamingResponse:
    n = random.randint(12, 18)
    data: dict = {}
    for i in range(1, 29):
        std = random.uniform(0.9, 1.4)
        col = np.round(np.random.normal(0.0, std, n), 6)
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
        headers={"Content-Disposition": 'attachment; filename="sample_transactions.csv"'},
    )


@router.get("/sample-csv", summary="Download sample CSV")
@router.get("/fraud/sample-csv", summary="Download sample CSV")
def sample_csv():
    return _generate_synthetic_csv()
