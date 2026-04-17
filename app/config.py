from __future__ import annotations

import os

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """App settings loaded from environment variables."""

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    APP_NAME: str = "Fraud Detection API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    MODEL_PATH: str = os.path.join("saved_models", "xgb_model.joblib")
    SCALER_PATH: str = os.path.join("saved_models", "amount_scaler.joblib")
    METADATA_PATH: str = os.path.join("saved_models", "model_metadata.json")

    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/fraud_detection"

    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = "logs"
    LOG_FILE: str = "app.log"

    # JWT
    SECRET_KEY: str = "change-me-in-production-use-a-real-secret"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # Default admin account (seeded on first startup)
    DEFAULT_ADMIN_USERNAME: str = "admin"
    DEFAULT_ADMIN_EMAIL: str = "admin@fraudapi.local"
    DEFAULT_ADMIN_PASSWORD: str = "admin123"


settings = Settings()
