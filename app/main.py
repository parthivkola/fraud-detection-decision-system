from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.schemas import HealthResponse
from ml.utils import load_artifact, load_json


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.database import engine
    from app.models import Base, ModelVersion, User
    from app.auth import hash_password
    from sqlalchemy.orm import Session

    # Create tables
    Base.metadata.create_all(bind=engine)

    with Session(engine) as db:
        # Seed admin
        if not db.query(User).filter(User.username == settings.DEFAULT_ADMIN_USERNAME).first():
            db.add(User(
                username=settings.DEFAULT_ADMIN_USERNAME,
                email=settings.DEFAULT_ADMIN_EMAIL,
                hashed_password=hash_password(settings.DEFAULT_ADMIN_PASSWORD),
                role="admin",
            ))
            db.commit()

        # Seed model versions
        if db.query(ModelVersion).count() == 0:
            db.add_all([
                ModelVersion(
                    version_tag="v1-champion",
                    description="Standard threshold (0.90). Balanced precision and recall.",
                    file_path="saved_models/v1/xgb_model.joblib",
                    scaler_path="saved_models/v1/amount_scaler.joblib",
                    metadata_path="saved_models/v1/model_metadata.json",
                    is_active=True,
                    ab_weight=0.6,
                ),
                ModelVersion(
                    version_tag="v2-high-recall",
                    description="Low threshold (0.65). Catches more fraud, more reviews.",
                    file_path="saved_models/v2/xgb_model.joblib",
                    scaler_path="saved_models/v2/amount_scaler.joblib",
                    metadata_path="saved_models/v2/model_metadata.json",
                    is_active=True,
                    ab_weight=0.3,
                ),
                ModelVersion(
                    version_tag="v3-high-precision",
                    description="High threshold (0.95). Minimises false positives.",
                    file_path="saved_models/v3/xgb_model.joblib",
                    scaler_path="saved_models/v3/amount_scaler.joblib",
                    metadata_path="saved_models/v3/model_metadata.json",
                    is_active=False,
                    ab_weight=0.1,
                ),
            ])
            db.commit()

        # Load all model versions into memory
        loaded: dict = {}
        for v in db.query(ModelVersion).all():
            try:
                md = load_json(v.metadata_path)
                loaded[v.version_tag] = {
                    "model": load_artifact(v.file_path),
                    "scaler": load_artifact(v.scaler_path),
                    "threshold": md["threshold"],
                    "ab_weight": v.ab_weight,
                    "is_active": v.is_active,
                }
            except Exception as exc:
                print(f"[WARN] Could not load {v.version_tag}: {exc}")

        app.state.models = loaded

    # Also expose the default/legacy model
    try:
        app.state.model = load_artifact(settings.MODEL_PATH)
        app.state.scaler = load_artifact(settings.SCALER_PATH)
        meta = load_json(settings.METADATA_PATH)
        app.state.threshold = meta["threshold"]
    except Exception as exc:
        print(f"[WARN] Could not load root default model: {exc}")
        app.state.model = None
        app.state.scaler = None
        app.state.threshold = 0.90

    app.state.startup_time = time.time()
    yield


app = FastAPI(
    title="Fraud Detection API",
    version="2.0.0",
    description="XGBoost-based fraud detection with model versioning and A/B testing.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
from app.routers.auth import router as auth_router
from app.routers.models import router as models_router
from app.routers.fraud import router as fraud_router
from app.routers.metrics import router as metrics_router
from app.routers.sample import router as sample_router

app.include_router(auth_router)
app.include_router(models_router)
app.include_router(fraud_router)
app.include_router(metrics_router)
app.include_router(sample_router)

# Also expose sample on /api/predict/sample-csv and /api/sample-csv for convenience
@app.get("/api/predict/sample-csv", tags=["sample"])
@app.get("/api/sample-csv", tags=["sample"])
def extra_sample_csv():
    from app.routers.sample import _generate_synthetic_csv
    return _generate_synthetic_csv()


DIST = Path("frontend/dist")
if DIST.exists():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")


@app.get("/", response_model=HealthResponse, tags=["health"])
async def root(request: Request):
    # Serve React UI if browser requests HTML
    accept = request.headers.get("accept", "")
    if "text/html" in accept and DIST.exists() and (DIST / "index.html").exists():
        return FileResponse(DIST / "index.html")
    return {"status": "ok", "version": app.version, "model_loaded": app.state.model is not None}


@app.get("/health", response_model=HealthResponse, tags=["health"])
@app.get("/api/health", response_model=HealthResponse, tags=["health"])
@app.get("/api/v1/health", response_model=HealthResponse, tags=["health"])
def health():
    return {"status": "ok", "version": app.version, "model_loaded": app.state.model is not None}


@app.get("/{full_path:path}", include_in_schema=False)
async def spa_catch_all(full_path: str, request: Request):
    if DIST.exists() and (DIST / "index.html").exists():
        if not full_path.startswith("api"):
            return FileResponse(DIST / "index.html")
    return JSONResponse(status_code=404, content={"detail": "Not Found"})
