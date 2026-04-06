"""
FastAPI application entry point.

Loads ML artifacts (model, scaler, metadata) at startup via lifespan.
Creates database tables if they don't exist.
Includes the fraud router. Auth and DB routers can be added later.
"""

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
    """Load ML artifacts and create DB tables at startup."""

    # ---- Database tables ----
    # Import here (not at top) to avoid circular imports.
    # Base.metadata.create_all() looks at every model that inherits from
    # Base (PredictionBatch, PredictionResult) and creates the
    # corresponding tables in PostgreSQL.  If the tables already exist,
    # this is a no-op — safe to run every time.
    from app.database import engine
    from app.models import Base  # noqa: F401  (ensures models are registered)

    logger.info("Creating database tables (if they don't exist)...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ready.")

    # ---- ML artifacts ----
    logger.info("Loading ML artifacts...")

    # Load model + scaler
    app.state.model = load_artifact(settings.MODEL_PATH)
    app.state.scaler = load_artifact(settings.SCALER_PATH)

    # Load threshold from metadata
    metadata = load_json(settings.METADATA_PATH)
    app.state.threshold = metadata["threshold"]
    app.state.model_features = metadata["features"]

    logger.info(
        f"Model loaded: threshold={app.state.threshold}, "
        f"features={len(app.state.model_features)}"
    )

    yield  # App runs here

    # Cleanup (if needed in the future)
    logger.info("Shutting down...")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Fraud detection API powered by XGBoost",
    lifespan=lifespan,
)

# CORS — allow all for dev, tighten in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

from app.routers.fraud import router as fraud_router  # noqa: E402

app.include_router(fraud_router)

# Future:
# from app.routers.users import router as users_router
# from app.routers.model import router as model_router
# app.include_router(users_router)
# app.include_router(model_router)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Basic health check — confirms the API and model are loaded."""
    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        model_loaded=hasattr(app.state, "model") and app.state.model is not None,
    )
