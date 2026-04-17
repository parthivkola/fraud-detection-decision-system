"""Tests for auth: register, login, /me, role protection."""

from tests.conftest import auth_header


class TestRegister:
    def test_register_user(self, client):
        resp = client.post("/api/v1/auth/register", json={
            "username": "newuser",
            "email": "new@test.com",
            "password": "secret123",
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["username"] == "newuser"
        assert data["role"] == "analyst"
        assert "hashed_password" not in data

    def test_register_duplicate_username(self, client):
        payload = {
            "username": "dupeuser",
            "email": "dupe1@test.com",
            "password": "secret123",
        }
        client.post("/api/v1/auth/register", json=payload)

        resp = client.post("/api/v1/auth/register", json={
            "username": "dupeuser",
            "email": "dupe2@test.com",
            "password": "secret123",
        })
        assert resp.status_code == 400
        assert "already taken" in resp.json()["detail"]

    def test_register_duplicate_email(self, client):
        client.post("/api/v1/auth/register", json={
            "username": "user1",
            "email": "same@test.com",
            "password": "secret123",
        })
        resp = client.post("/api/v1/auth/register", json={
            "username": "user2",
            "email": "same@test.com",
            "password": "secret123",
        })
        assert resp.status_code == 400
        assert "already registered" in resp.json()["detail"]


class TestLogin:
    def test_login_success(self, client):
        client.post("/api/v1/auth/register", json={
            "username": "loginuser",
            "email": "login@test.com",
            "password": "secret123",
        })
        resp = client.post("/api/v1/auth/login", json={
            "username": "loginuser",
            "email": "login@test.com",
            "password": "secret123",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_login_wrong_password(self, client):
        client.post("/api/v1/auth/register", json={
            "username": "loginuser2",
            "email": "login2@test.com",
            "password": "secret123",
        })
        resp = client.post("/api/v1/auth/login", json={
            "username": "loginuser2",
            "email": "login2@test.com",
            "password": "wrongpass",
        })
        assert resp.status_code == 401


class TestProtected:
    def test_me_with_token(self, client, analyst_token):
        resp = client.get("/api/v1/auth/me", headers=auth_header(analyst_token))
        assert resp.status_code == 200
        assert resp.json()["username"] == "testanalyst"

    def test_protected_route_without_token(self, client):
        resp = client.get("/api/v1/auth/me")
        assert resp.status_code == 401
