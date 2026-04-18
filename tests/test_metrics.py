"""Tests for the /metrics endpoint."""

from tests.conftest import auth_header


class TestMetrics:
    def test_metrics_endpoint(self, client, admin_token):
        resp = client.get(
            "/api/v1/metrics",
            headers=auth_header(admin_token),
        )
        assert resp.status_code == 200
        data = resp.json()
        # Operational
        assert "total_predictions" in data
        assert "total_batches" in data
        assert "uptime_seconds" in data
        assert "active_model_versions" in data
        assert "threshold" in data
        # Detection stats
        assert "flagged_fraud" in data
        assert "flagged_legitimate" in data
        assert "fraud_flag_rate" in data
        # Model quality
        assert "model_precision" in data
        assert "model_recall" in data
        assert "model_f1" in data
        assert "model_roc_auc" in data
        # Risk
        assert "risk_distribution" in data
        assert data["uptime_seconds"] >= 0

    def test_metrics_allows_analyst(self, client, analyst_token):
        resp = client.get(
            "/api/v1/metrics",
            headers=auth_header(analyst_token),
        )
        assert resp.status_code == 200

    def test_metrics_requires_auth(self, client):
        resp = client.get("/api/v1/metrics")
        assert resp.status_code == 401
