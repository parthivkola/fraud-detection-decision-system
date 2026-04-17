"""Tests for model versioning endpoints."""

from tests.conftest import auth_header


class TestModelVersions:
    def test_register_model_version(self, client, admin_token):
        resp = client.post(
            "/api/v1/models/",
            json={
                "version_tag": "v1.0",
                "description": "Initial model",
                "file_path": "saved_models/xgb_model.joblib",
                "scaler_path": "saved_models/amount_scaler.joblib",
                "metadata_path": "saved_models/model_metadata.json",
            },
            headers=auth_header(admin_token),
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["version_tag"] == "v1.0"
        assert data["is_active"] is False

    def test_register_requires_admin(self, client, analyst_token):
        resp = client.post(
            "/api/v1/models/",
            json={
                "version_tag": "v1.0",
                "description": "Initial model",
                "file_path": "saved_models/xgb_model.joblib",
                "scaler_path": "saved_models/amount_scaler.joblib",
                "metadata_path": "saved_models/model_metadata.json",
            },
            headers=auth_header(analyst_token),
        )
        assert resp.status_code == 403

    def test_list_model_versions(self, client, admin_token):
        # Register one first
        client.post(
            "/api/v1/models/",
            json={
                "version_tag": "v2.0",
                "file_path": "saved_models/xgb_model.joblib",
                "scaler_path": "saved_models/amount_scaler.joblib",
                "metadata_path": "saved_models/model_metadata.json",
            },
            headers=auth_header(admin_token),
        )
        resp = client.get(
            "/api/v1/models/",
            headers=auth_header(admin_token),
        )
        assert resp.status_code == 200
        assert len(resp.json()) >= 1

    def test_activate_model(self, client, admin_token):
        create_resp = client.post(
            "/api/v1/models/",
            json={
                "version_tag": "v3.0",
                "file_path": "saved_models/xgb_model.joblib",
                "scaler_path": "saved_models/amount_scaler.joblib",
                "metadata_path": "saved_models/model_metadata.json",
            },
            headers=auth_header(admin_token),
        )
        version_id = create_resp.json()["id"]

        resp = client.patch(
            f"/api/v1/models/{version_id}/activate",
            headers=auth_header(admin_token),
        )
        assert resp.status_code == 200
        assert resp.json()["is_active"] is True

    def test_update_ab_weight(self, client, admin_token):
        create_resp = client.post(
            "/api/v1/models/",
            json={
                "version_tag": "v4.0",
                "file_path": "saved_models/xgb_model.joblib",
                "scaler_path": "saved_models/amount_scaler.joblib",
                "metadata_path": "saved_models/model_metadata.json",
            },
            headers=auth_header(admin_token),
        )
        version_id = create_resp.json()["id"]

        resp = client.patch(
            f"/api/v1/models/{version_id}",
            json={"ab_weight": 0.3},
            headers=auth_header(admin_token),
        )
        assert resp.status_code == 200
        assert resp.json()["ab_weight"] == 0.3
