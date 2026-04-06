"""
Application configuration.

Loads settings from environment variables with sensible defaults.
Threshold is loaded from model_metadata.json at startup.
"""

from __future__ import annotations

import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """App settings loaded from environment variables."""

    # --- API ---
    APP_NAME: str = "Fraud Detection API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # --- ML artifacts ---
    MODEL_PATH: str = os.path.join("saved_models", "xgb_model.joblib")
    SCALER_PATH: str = os.path.join("saved_models", "amount_scaler.joblib")
    METADATA_PATH: str = os.path.join("saved_models", "model_metadata.json")

    # --- Database ---
    DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/fraud_detection"

    # --- Future: Auth ---
    # SECRET_KEY: str = "changeme"
    # ALGORITHM: str = "HS256"
    # ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
