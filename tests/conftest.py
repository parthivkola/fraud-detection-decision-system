"""Shared fixtures for the test suite.

Uses an in-memory SQLite database so tests are fast, isolated, and don't
need a running PostgreSQL instance.
"""
from __future__ import annotations

from typing import Generator
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.auth import create_access_token, hash_password
from app.database import Base, get_db
from app.main import app
from app.models import User

# ── In-memory SQLite ──────────────────────────────────────────────────────────

SQLALCHEMY_TEST_URL = "sqlite:///./test.db"

engine = create_engine(SQLALCHEMY_TEST_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


@pytest.fixture(autouse=True)
def setup_database():
    """Create all tables before each test and drop them after."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture()
def db_session() -> Generator[Session, None, None]:
    """Yield a fresh database session."""
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


def _override_get_db():
    """Dependency override for get_db."""
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def mock_model():
    """Return a mock XGBoost model that always returns known probabilities."""
    model = MagicMock()
    model.predict_proba.return_value = np.array([
        [0.95, 0.05],
        [0.20, 0.80],
        [0.50, 0.50],
    ])
    return model


@pytest.fixture()
def client(mock_model) -> TestClient:
    """TestClient with DB override and mock ML model."""
    app.dependency_overrides[get_db] = _override_get_db

    # Set mock model on app state
    app.state.model = mock_model
    app.state.scaler = MagicMock()
    app.state.threshold = 0.5
    app.state.model_features = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    app.state.loaded_versions = {}
    import time
    app.state.startup_time = time.time()

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


# ── Helper: create users in DB and get tokens ─────────────────────────────────

@pytest.fixture()
def analyst_user(db_session: Session) -> User:
    """Create and return an analyst user."""
    user = User(
        username="testanalyst",
        email="analyst@test.com",
        hashed_password=hash_password("password123"),
        role="analyst",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture()
def admin_user(db_session: Session) -> User:
    """Create and return an admin user."""
    user = User(
        username="testadmin",
        email="admin@test.com",
        hashed_password=hash_password("adminpass123"),
        role="admin",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture()
def analyst_token(analyst_user: User) -> str:
    """Return a JWT for the analyst user."""
    return create_access_token(data={"sub": analyst_user.username, "role": analyst_user.role})


@pytest.fixture()
def admin_token(admin_user: User) -> str:
    """Return a JWT for the admin user."""
    return create_access_token(data={"sub": admin_user.username, "role": admin_user.role})


def auth_header(token: str) -> dict:
    """Convenience: return Authorization header dict."""
    return {"Authorization": f"Bearer {token}"}
