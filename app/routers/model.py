from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.auth import get_current_user, require_role
from app.database import get_db
from app.logger import logger
from app.models import ModelVersion, User
from app.schemas import ModelVersionCreate, ModelVersionOut, ModelVersionUpdate

router = APIRouter(prefix="/api/v1/models", tags=["models"])


@router.post(
    "/",
    response_model=ModelVersionOut,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new model version",
)
def register_model(
    payload: ModelVersionCreate,
    db: Session = Depends(get_db),
    admin: User = Depends(require_role("admin")),
):
    """Register a new model version. Admin only."""
    if db.query(ModelVersion).filter(ModelVersion.version_tag == payload.version_tag).first():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Version tag '{payload.version_tag}' already exists",
        )

    version = ModelVersion(
        version_tag=payload.version_tag,
        description=payload.description,
        file_path=payload.file_path,
        scaler_path=payload.scaler_path,
        metadata_path=payload.metadata_path,
    )
    db.add(version)
    db.commit()
    db.refresh(version)

    logger.info(f"Model version '{version.version_tag}' registered by {admin.username}")
    return version


@router.get(
    "/",
    response_model=List[ModelVersionOut],
    summary="List all model versions",
)
def list_models(
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """List all registered model versions, newest first."""
    return (
        db.query(ModelVersion)
        .order_by(ModelVersion.created_at.desc())
        .all()
    )


@router.get(
    "/{version_id}",
    response_model=ModelVersionOut,
    summary="Get a specific model version",
)
def get_model(
    version_id: int,
    db: Session = Depends(get_db),
    _user: User = Depends(get_current_user),
):
    """Get details of a single model version."""
    version = db.query(ModelVersion).filter(ModelVersion.id == version_id).first()
    if not version:
        raise HTTPException(status_code=404, detail="Model version not found")
    return version


@router.patch(
    "/{version_id}",
    response_model=ModelVersionOut,
    summary="Update a model version",
)
def update_model(
    version_id: int,
    payload: ModelVersionUpdate,
    db: Session = Depends(get_db),
    admin: User = Depends(require_role("admin")),
):
    """Update model description or A/B weight. Admin only."""
    version = db.query(ModelVersion).filter(ModelVersion.id == version_id).first()
    if not version:
        raise HTTPException(status_code=404, detail="Model version not found")

    if payload.description is not None:
        version.description = payload.description
    if payload.ab_weight is not None:
        version.ab_weight = payload.ab_weight

    db.commit()
    db.refresh(version)

    logger.info(f"Model version '{version.version_tag}' updated by {admin.username}")
    return version


@router.patch(
    "/{version_id}/activate",
    response_model=ModelVersionOut,
    summary="Activate a model version",
)
def activate_model(
    version_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(require_role("admin")),
):
    """Set a model version as active. Does NOT deactivate others (supports A/B)."""
    version = db.query(ModelVersion).filter(ModelVersion.id == version_id).first()
    if not version:
        raise HTTPException(status_code=404, detail="Model version not found")

    version.is_active = True
    db.commit()
    db.refresh(version)

    logger.info(
        f"Model version '{version.version_tag}' activated by {admin.username}"
    )
    return version


@router.patch(
    "/{version_id}/deactivate",
    response_model=ModelVersionOut,
    summary="Deactivate a model version",
)
def deactivate_model(
    version_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(require_role("admin")),
):
    """Deactivate a model version."""
    version = db.query(ModelVersion).filter(ModelVersion.id == version_id).first()
    if not version:
        raise HTTPException(status_code=404, detail="Model version not found")

    version.is_active = False
    db.commit()
    db.refresh(version)

    logger.info(
        f"Model version '{version.version_tag}' deactivated by {admin.username}"
    )
    return version
