from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from app.auth import get_current_user, require_role
from app.database import get_db
from app.models import ModelVersion, User
from app.schemas import (
    ModelVersionCreate,
    ModelVersionOut,
    ModelVersionUpdate,
    UpdateWeightRequest,
)

router = APIRouter(prefix="/api/v1/models", tags=["models"])


@router.post("", response_model=ModelVersionOut, status_code=201)
@router.post("/", response_model=ModelVersionOut, status_code=201)
def create_model(
    body: ModelVersionCreate,
    request: Request,
    db: Session = Depends(get_db),
    _: User = Depends(require_role("admin")),
):
    if db.query(ModelVersion).filter(ModelVersion.version_tag == body.version_tag).first():
        raise HTTPException(400, f"Version tag '{body.version_tag}' already exists")
    v = ModelVersion(
        version_tag=body.version_tag,
        description=body.description,
        file_path=body.file_path,
        scaler_path=body.scaler_path,
        metadata_path=body.metadata_path,
        is_active=False,
        ab_weight=1.0,
    )
    db.add(v)
    db.commit()
    db.refresh(v)
    return v


@router.get("", response_model=list[ModelVersionOut])
@router.get("/", response_model=list[ModelVersionOut])
def list_models(db: Session = Depends(get_db), _: User = Depends(get_current_user)):
    return db.query(ModelVersion).order_by(ModelVersion.created_at).all()


@router.patch("/{model_id}/activate", response_model=ModelVersionOut)
def activate(
    model_id: int,
    request: Request,
    db: Session = Depends(get_db),
    _: User = Depends(require_role("admin")),
):
    v = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()
    if not v:
        raise HTTPException(404, "Model not found")
    v.is_active = True
    db.commit()
    db.refresh(v)
    lv = getattr(request.app.state, "models", {})
    if v.version_tag in lv:
        lv[v.version_tag]["is_active"] = True
    return v


@router.patch("/{model_id}/deactivate", response_model=ModelVersionOut)
def deactivate(
    model_id: int,
    request: Request,
    db: Session = Depends(get_db),
    _: User = Depends(require_role("admin")),
):
    v = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()
    if not v:
        raise HTTPException(404, "Model not found")
    v.is_active = False
    db.commit()
    db.refresh(v)
    lv = getattr(request.app.state, "models", {})
    if v.version_tag in lv:
        lv[v.version_tag]["is_active"] = False
    return v


@router.patch("/{model_id}/weight", response_model=ModelVersionOut)
def set_weight(
    model_id: int,
    body: UpdateWeightRequest,
    request: Request,
    db: Session = Depends(get_db),
    _: User = Depends(require_role("admin")),
):
    if not 0.0 <= body.ab_weight <= 1.0:
        raise HTTPException(400, "Weight must be between 0.0 and 1.0")
    v = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()
    if not v:
        raise HTTPException(404, "Model not found")
    v.ab_weight = body.ab_weight
    db.commit()
    db.refresh(v)
    lv = getattr(request.app.state, "models", {})
    if v.version_tag in lv:
        lv[v.version_tag]["ab_weight"] = body.ab_weight
    return v


@router.patch("/{model_id}", response_model=ModelVersionOut)
def update_model(
    model_id: int,
    body: ModelVersionUpdate,
    request: Request,
    db: Session = Depends(get_db),
    _: User = Depends(require_role("admin")),
):
    v = db.query(ModelVersion).filter(ModelVersion.id == model_id).first()
    if not v:
        raise HTTPException(404, "Model not found")
    if body.description is not None:
        v.description = body.description
    if body.ab_weight is not None:
        if not 0.0 <= body.ab_weight <= 1.0:
            raise HTTPException(400, "Weight must be between 0.0 and 1.0")
        v.ab_weight = body.ab_weight
    if body.is_active is not None:
        v.is_active = body.is_active
    db.commit()
    db.refresh(v)
    lv = getattr(request.app.state, "models", {})
    if v.version_tag in lv:
        if body.ab_weight is not None:
            lv[v.version_tag]["ab_weight"] = body.ab_weight
        if body.is_active is not None:
            lv[v.version_tag]["is_active"] = body.is_active
    return v
