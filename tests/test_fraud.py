"""Tests for fraud prediction and history endpoints."""

import io
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

from tests.conftest import auth_header

FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]


def _make_csv(n_rows: int = 3) -> bytes:
    """Create a minimal valid CSV in memory."""
    data = {col: [0.1] * n_rows for col in FEATURE_COLS}
    df = pd.DataFrame(data)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def _make_transformed_df(n_rows: int = 3) -> pd.DataFrame:
    """Create a DataFrame with correct feature columns matching the model."""
    data = {col: [0.1] * n_rows for col in FEATURE_COLS}
    return pd.DataFrame(data)


class TestPredict:
    @patch("app.routers.fraud.transform_new_data")
    def test_predict_valid_csv(self, mock_transform, client, analyst_token):
        mock_transform.return_value = _make_transformed_df(3)
        csv_bytes = _make_csv(3)
        resp = client.post(
            "/api/v1/fraud/predict",
            files={"file": ("test.csv", csv_bytes, "text/csv")},
            headers=auth_header(analyst_token),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "batch_id" in data
        assert len(data["predictions"]) == 3
        assert data["summary"]["total_transactions"] == 3

    def test_predict_invalid_file_type(self, client, analyst_token):
        resp = client.post(
            "/api/v1/fraud/predict",
            files={"file": ("test.txt", b"not a csv", "text/plain")},
            headers=auth_header(analyst_token),
        )
        assert resp.status_code == 400
        assert "CSV" in resp.json()["detail"]

    def test_predict_missing_columns(self, client, analyst_token):
        df = pd.DataFrame({"V1": [0.1], "Amount": [100.0]})  # Missing V2-V28
        buf = io.BytesIO()
        df.to_csv(buf, index=False)
        csv_bytes = buf.getvalue()

        resp = client.post(
            "/api/v1/fraud/predict",
            files={"file": ("test.csv", csv_bytes, "text/csv")},
            headers=auth_header(analyst_token),
        )
        assert resp.status_code == 400
        assert "Missing required columns" in resp.json()["detail"]

    def test_predict_requires_auth(self, client):
        csv_bytes = _make_csv(1)
        resp = client.post(
            "/api/v1/fraud/predict",
            files={"file": ("test.csv", csv_bytes, "text/csv")},
        )
        assert resp.status_code == 401


class TestHistory:
    @patch("app.routers.fraud.transform_new_data")
    def test_history_list(self, mock_transform, client, analyst_token):
        mock_transform.return_value = _make_transformed_df(3)
        # Create a prediction first
        csv_bytes = _make_csv(3)
        create_resp = client.post(
            "/api/v1/fraud/predict",
            files={"file": ("test.csv", csv_bytes, "text/csv")},
            headers=auth_header(analyst_token),
        )
        assert create_resp.status_code == 200, f"Predict failed: {create_resp.json()}"

        resp = client.get(
            "/api/v1/fraud/history",
            headers=auth_header(analyst_token),
        )
        assert resp.status_code == 200
        assert len(resp.json()) >= 1

    def test_history_detail_not_found(self, client, analyst_token):
        resp = client.get(
            "/api/v1/fraud/history/99999",
            headers=auth_header(analyst_token),
        )
        assert resp.status_code == 404
