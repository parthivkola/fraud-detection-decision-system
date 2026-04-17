# Fraud Detection Decision System

A production-grade **FastAPI** backend for real-time credit card fraud detection, powered by **XGBoost**. Features JWT authentication with role-based access, model versioning with A/B testing, structured JSON logging, and full Docker support.

---

## Architecture

```
fraud-detection-decision-system/
├── app/                        # FastAPI application
│   ├── main.py                 # App entry point, lifespan, router registration
│   ├── config.py               # Pydantic settings (env vars)
│   ├── database.py             # SQLAlchemy engine + session
│   ├── models.py               # ORM models (User, ModelVersion, PredictionBatch, PredictionResult)
│   ├── schemas.py              # Pydantic request/response schemas
│   ├── auth.py                 # JWT + bcrypt auth utilities + FastAPI deps
│   ├── crud.py                 # Database CRUD operations
│   ├── risk.py                 # Risk assessment engine (LOW → CRITICAL)
│   ├── logger.py               # Structured JSON logging setup
│   └── routers/
│       ├── fraud.py            # POST /predict, GET /history
│       ├── users.py            # POST /register, POST /login, GET /me
│       ├── model.py            # Model version CRUD + activate/deactivate
│       └── metrics.py          # GET /metrics (admin only)
├── ml/                         # ML pipeline
│   ├── preprocessing/
│   │   ├── prepare_data.py     # Data loading, validation, cleaning
│   │   └── transform_features.py  # log1p + StandardScaler on Amount
│   ├── training/
│   │   ├── train.py            # XGBoost training pipeline
│   │   └── evaluate.py         # Threshold tuning + evaluation metrics
│   └── utils.py                # Artifact save/load utilities
├── saved_models/               # Serialized model + scaler + metadata
├── alembic/                    # Database migrations
├── tests/                      # 29 pytest tests
├── Dockerfile                  # Multi-stage Docker build
├── docker-compose.yml          # API + PostgreSQL stack
└── requirements.txt
```

---

## Quick Start

### Local Development

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up PostgreSQL and configure .env
cp .env.example .env  # Edit DATABASE_URL, SECRET_KEY

# 4. Run database migrations
alembic upgrade head

# 5. Start the server
uvicorn app.main:app --reload
```

### Docker

```bash
docker compose up --build -d
# API at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

---

## API Endpoints

### Health
| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/` | None | Health check |
| GET | `/health` | None | Health check (alias) |

### Authentication
| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/v1/auth/register` | None | Create a new user account |
| POST | `/api/v1/auth/login` | None | Get JWT token |
| GET | `/api/v1/auth/me` | Bearer | Get current user profile |
| PATCH | `/api/v1/auth/users/{id}/role` | Admin | Change user role |

### Fraud Detection
| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/v1/fraud/predict` | Bearer | Upload CSV → get fraud predictions |
| GET | `/api/v1/fraud/history` | Bearer | List prediction batches |
| GET | `/api/v1/fraud/history/{id}` | Bearer | Get batch with all results |

### Model Versioning
| Method | Path | Auth | Description |
|--------|------|------|-------------|
| POST | `/api/v1/models/` | Admin | Register a new model version |
| GET | `/api/v1/models/` | Bearer | List all model versions |
| GET | `/api/v1/models/{id}` | Bearer | Get model version details |
| PATCH | `/api/v1/models/{id}` | Admin | Update description / A/B weight |
| PATCH | `/api/v1/models/{id}/activate` | Admin | Activate for serving |
| PATCH | `/api/v1/models/{id}/deactivate` | Admin | Deactivate |

### Metrics
| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/api/v1/metrics` | Admin | System metrics & statistics |

---

## Authentication Flow

1. **Register** → `POST /api/v1/auth/register` with `{username, email, password}`
2. **Login** → `POST /api/v1/auth/login` → returns `{access_token, token_type}`
3. **Use token** → pass `Authorization: Bearer <token>` header on all protected endpoints

### Roles
- **analyst** (default) — can run predictions, view history, list models
- **admin** — all analyst permissions + manage models, view metrics, change user roles

A default admin account (`admin / admin123`) is seeded automatically on first startup.

---

## Model Versioning & A/B Testing

Register multiple model versions pointing to different serialized artifacts:

```bash
# Register v2.0
curl -X POST http://localhost:8000/api/v1/models/ \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"version_tag": "v2.0", "file_path": "saved_models/v2/model.joblib", ...}'

# Activate both v1.0 and v2.0 with different weights for A/B testing
curl -X PATCH http://localhost:8000/api/v1/models/1/activate -H "Authorization: Bearer $ADMIN_TOKEN"
curl -X PATCH http://localhost:8000/api/v1/models/2/activate -H "Authorization: Bearer $ADMIN_TOKEN"
curl -X PATCH http://localhost:8000/api/v1/models/1 \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"ab_weight": 0.7}'  # 70% traffic
curl -X PATCH http://localhost:8000/api/v1/models/2 \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"ab_weight": 0.3}'  # 30% traffic
```

Each prediction batch records which model version produced it, enabling performance comparison.

---

## Risk Assessment

| Probability Range | Risk Level | Decision |
|-------------------|------------|----------|
| < 0.3 | LOW | approve |
| 0.3 – 0.6 | MEDIUM | review |
| 0.6 – threshold | HIGH | review |
| ≥ threshold | CRITICAL | block |

---

## Database Migrations

```bash
# Generate a new migration after model changes
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback one step
alembic downgrade -1
```

---

## Logging

All logs are written as **structured JSON** to `logs/app.log`:

```json
{
  "timestamp": 1713345600.123,
  "level": "INFO",
  "logger": "fraud_api",
  "message": "Prediction complete: 100 txns, 3 flagged, batch_id=42",
  "module": "fraud",
  "function": "predict_fraud",
  "line": 132
}
```

Console output is human-readable when `DEBUG=true`, JSON otherwise.

---

## Testing

```bash
# Run all 29 tests
pytest tests/ -v

# Run specific test file
pytest tests/test_auth.py -v

# Run with coverage
pytest tests/ --cov=app --cov-report=term-missing
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/fraud_detection` | PostgreSQL connection string |
| `SECRET_KEY` | `change-me-...` | JWT signing key |
| `ALGORITHM` | `HS256` | JWT algorithm |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `30` | Token TTL |
| `LOG_LEVEL` | `INFO` | Logging level |
| `DEBUG` | `false` | Enable debug mode |

---

## ML Pipeline

The model is trained on the [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) dataset.

```bash
# Train model (requires data/raw/creditcard.csv)
python -m ml.training.train
```

### Pipeline Steps
1. Load data → validate columns → deduplicate → drop `Time`
2. Split into train (70%) / validation (15%) / test (15%) — stratified
3. Apply `log1p` + `StandardScaler` on `Amount`
4. Train XGBoost with class-weight balancing
5. Tune threshold on validation set (maximize recall with precision ≥ 0.80)
6. Evaluate on test set → save model, scaler, metadata

### Model Performance
| Metric | Value |
|--------|-------|
| ROC-AUC | 0.970 |
| PR-AUC | 0.781 |
| Precision | 0.864 |
| Recall | 0.803 |
| F1 | 0.832 |
