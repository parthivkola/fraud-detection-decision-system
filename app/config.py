from __future__ import annotations

import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """App settings loaded from environment variables."""

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

    # SECRET_KEY: str = "changeme"
    # ALGORITHM: str = "HS256"
    # ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
