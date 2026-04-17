from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.logger import logger
from app.schemas import HealthResponse
from ml.utils import load_artifact, load_json


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create DB tables, seed admin, and load ML artifacts at startup."""

    # Imported here to avoid circular imports
    from app.database import engine
    from app.models import Base, ModelVersion, User  # noqa: F401

    # ── Database setup ────────────────────────────────────────────────────
    logger.info("Creating database tables (if they don't exist)...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ready.")

    # ── Seed default admin ────────────────────────────────────────────────
    from sqlalchemy.orm import Session
    from app.auth import hash_password

    with Session(engine) as db:
        existing = db.query(User).filter(User.username == settings.DEFAULT_ADMIN_USERNAME).first()
        if not existing:
            admin = User(
                username=settings.DEFAULT_ADMIN_USERNAME,
                email=settings.DEFAULT_ADMIN_EMAIL,
                hashed_password=hash_password(settings.DEFAULT_ADMIN_PASSWORD),
                role="admin",
            )
            db.add(admin)
            db.commit()
            logger.info(f"Default admin user '{settings.DEFAULT_ADMIN_USERNAME}' created.")
        else:
            logger.info("Default admin user already exists, skipping seed.")

    # ── Load default ML artifacts ─────────────────────────────────────────
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

    # ── Load versioned models (if any are active in DB) ───────────────────
    with Session(engine) as db:
        active_versions = db.query(ModelVersion).filter(ModelVersion.is_active.is_(True)).all()
        loaded_versions = {}
        for v in active_versions:
            try:
                m = load_artifact(v.file_path)
                s = load_artifact(v.scaler_path)
                md = load_json(v.metadata_path)
                loaded_versions[v.version_tag] = {
                    "model": m,
                    "scaler": s,
                    "threshold": md["threshold"],
                    "ab_weight": v.ab_weight,
                    "version_id": v.id,
                }
                logger.info(f"Loaded model version '{v.version_tag}' (weight={v.ab_weight})")
            except Exception as e:
                logger.error(f"Failed to load model version '{v.version_tag}': {e}")

        app.state.loaded_versions = loaded_versions

    # ── Startup time for uptime metric ────────────────────────────────────
    app.state.startup_time = time.time()

    logger.info("Application startup complete.")

    yield

    logger.info("Shutting down...")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Fraud detection API powered by XGBoost with JWT auth, model versioning, and A/B testing.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Register routers ─────────────────────────────────────────────────────────

from app.routers.fraud import router as fraud_router       # noqa: E402
from app.routers.users import router as users_router       # noqa: E402
from app.routers.model import router as model_router       # noqa: E402
from app.routers.metrics import router as metrics_router   # noqa: E402
from app.routers.sample import router as sample_router     # noqa: E402

app.include_router(fraud_router)
app.include_router(users_router)
app.include_router(model_router)
app.include_router(metrics_router)
app.include_router(sample_router)


# ── Health endpoints ──────────────────────────────────────────────────────────

@app.get("/", response_model=HealthResponse, tags=["health"])
@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Basic health check — no auth required."""
    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        model_loaded=hasattr(app.state, "model") and app.state.model is not None,
    )


# ── Frontend ──────────────────────────────────────────────────────────────────

@app.get("/dashboard", include_in_schema=False)
async def serve_dashboard():
    """Serve the frontend dashboard."""
    return FileResponse("frontend/index.html")


app.mount("/static", StaticFiles(directory="frontend"), name="frontend")

