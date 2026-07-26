from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
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

    # ── Seed default showcase models if none exist ────────────────────────
    with Session(engine) as db:
        if db.query(ModelVersion).count() == 0:
            logger.info("Seeding default showcase model versions...")
            v1 = ModelVersion(
                version_tag="v1.0-champion",
                description="Baseline XGBoost champion model with standard decision threshold (0.90). Balanced precision and recall for daily transaction serving.",
                file_path="saved_models/v1/xgb_model.joblib",
                scaler_path="saved_models/v1/amount_scaler.joblib",
                metadata_path="saved_models/v1/model_metadata.json",
                is_active=True,
                ab_weight=0.6,
            )
            v2 = ModelVersion(
                version_tag="v2.0-recall-challenger",
                description="Tuned XGBoost challenger model optimized for maximum recall (0.65 decision threshold). Designed to catch emerging fraud patterns during peak hours.",
                file_path="saved_models/v2/xgb_model.joblib",
                scaler_path="saved_models/v2/amount_scaler.joblib",
                metadata_path="saved_models/v2/model_metadata.json",
                is_active=True,
                ab_weight=0.3,
            )
            v3 = ModelVersion(
                version_tag="v3.0-precision-guard",
                description="Conservative high-precision model minimizing false alarms (0.95 decision threshold). Recommended for VIP customer segments to avoid friction.",
                file_path="saved_models/v3/xgb_model.joblib",
                scaler_path="saved_models/v3/amount_scaler.joblib",
                metadata_path="saved_models/v3/model_metadata.json",
                is_active=False,
                ab_weight=0.1,
            )
            db.add_all([v1, v2, v3])
            db.commit()
            logger.info("Showcase model versions seeded successfully.")

        # ── Load versioned models ─────────────────────────────────────────────
        all_versions = db.query(ModelVersion).all()
        loaded_versions = {}
        for v in all_versions:
            try:
                m = load_artifact(v.file_path)
                s = load_artifact(v.scaler_path)
                md = load_json(v.metadata_path)
                loaded_versions[v.version_tag] = {
                    "model": m,
                    "scaler": s,
                    "threshold": md["threshold"],
                    "ab_weight": v.ab_weight,
                    "is_active": v.is_active,
                    "version_id": v.id,
                }
                logger.info(f"Loaded model version '{v.version_tag}' (active={v.is_active}, weight={v.ab_weight})")
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

@app.get("/", response_model=HealthResponse, tags=["health"], summary="Health check or Dashboard UI")
@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check(request: Request):
    """Basic health check (JSON), or serve dashboard UI if text/html requested."""
    accept = request.headers.get("accept", "")
    if "text/html" in accept:
        return FileResponse("frontend/index.html")
    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        model_loaded=hasattr(request.app.state, "model") and request.app.state.model is not None,
    )


# ── Frontend ──────────────────────────────────────────────────────────────────

@app.get("/dashboard", include_in_schema=False)
async def serve_dashboard():
    """Serve the frontend dashboard."""
    return FileResponse("frontend/index.html")


app.mount("/static", StaticFiles(directory="frontend"), name="frontend")

