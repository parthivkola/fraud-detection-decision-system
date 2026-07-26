from __future__ import annotations

import io
import random

import numpy as np
import pandas as pd
from fastapi import APIRouter
from fastapi.responses import Response, StreamingResponse

router = APIRouter(prefix="/api/v1", tags=["sample"])


def _generate_synthetic_csv() -> Response:
    n = random.randint(20, 25)
    data: dict = {}
    for i in range(1, 29):
        std = random.uniform(0.8, 1.2)
        data[f"V{i}"] = np.round(np.random.normal(0.0, std, n), 6)

    # Distribute: ~35% LOW, ~35% MEDIUM, ~30% CRITICAL
    n_crit = int(n * 0.30)
    n_med = int(n * 0.35)

    indices = list(range(n))
    random.shuffle(indices)
    crit_idx = indices[:n_crit]
    med_idx = indices[n_crit : n_crit + n_med]

    # Inject CRITICAL anomalies (strong negative/positive spikes on key fraud features)
    for idx in crit_idx:
        for feat in ["V1", "V3", "V4", "V7", "V10", "V12", "V14", "V17"]:
            data[feat][idx] = round(
                random.uniform(-16.0, -8.0) if feat != "V4" else random.uniform(8.0, 15.0), 6
            )

    # Inject MEDIUM anomalies (moderate spikes to trigger review queue without instant block)
    for idx in med_idx:
        for feat in ["V3", "V4", "V10", "V11", "V14"]:
            data[feat][idx] = round(
                random.uniform(-4.5, -2.5)
                if feat not in ["V4", "V11"]
                else random.uniform(2.5, 4.5),
                6,
            )

    amounts = []
    for idx in range(n):
        if idx in crit_idx:
            amounts.append(round(random.uniform(400.0, 2500.0), 2))
        elif idx in med_idx:
            amounts.append(round(random.uniform(120.0, 450.0), 2))
        else:
            amounts.append(round(random.uniform(5.0, 75.0), 2))
    data["Amount"] = amounts

    buf = io.StringIO()
    pd.DataFrame(data).to_csv(buf, index=False)
    content = buf.getvalue().encode("utf-8")
    return Response(
        content=content,
        media_type="text/csv",
        headers={
            "Content-Disposition": 'attachment; filename="sample_transactions.csv"',
            "Content-Length": str(len(content)),
        },
    )


@router.get("/sample-csv", summary="Download sample CSV")
@router.get("/fraud/sample-csv", summary="Download sample CSV")
def sample_csv():
    return _generate_synthetic_csv()
