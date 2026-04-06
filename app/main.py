from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.logger import logger
from app.schemas import HealthResponse
from ml.utils import load_artifact, load_json


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create DB tables and load ML artifacts at startup."""

    # Imported here to avoid circular imports
    from app.database import engine
    from app.models import Base  # noqa: F401

    logger.info("Creating database tables (if they don't exist)...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ready.")

    logger.info("Loading ML artifacts...")
    app.state.model = load_artifact(settings.MODEL_PATH)
    app.state.scaler = load_artifact(settings.SCALER_PATH)

    metadata = load_json(settings.METADATA_PATH)
    app.state.threshold = metadata["threshold"]
    app.state.model_features = metadata["features"]

    logger.info(
        f"Model loaded: threshold={app.state.threshold}, "
        f"features={len(app.state.model_features)}"
    )

    yield

    logger.info("Shutting down...")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Fraud detection API powered by XGBoost",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.routers.fraud import router as fraud_router  # noqa: E402

app.include_router(fraud_router)


@app.get("/", response_model=HealthResponse, tags=["health"])
async def health_check():
    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        model_loaded=hasattr(app.state, "model") and app.state.model is not None,
    )
