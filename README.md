# Credit Card Fraud Detection System

A professional, production-grade **FastAPI + React 18 (Vite + TypeScript)** full-stack platform for real-time credit card fraud inference, powered by **XGBoost gradient boosted trees**. Features JWT authentication with role-based access control, dynamic ML model registry with weighted A/B testing, real-time system analytics, and multi-stage Docker deployment support.

---

## Architecture Overview

```
fraud-detection-decision-system/
├── app/                        # FastAPI Backend Application
│   ├── main.py                 # App entry point, lifespan DB seed, React SPA mounting
│   ├── config.py               # Pydantic settings & environment configuration
│   ├── database.py             # SQLAlchemy engine & session management
│   ├── models.py               # ORM domain models (User, ModelVersion, PredictionBatch, PredictionResult)
│   ├── schemas.py              # Strict Pydantic request/response schemas
│   ├── auth.py                 # JWT & bcrypt security utilities
│   ├── risk.py                 # Risk assessment engine (LOW / MEDIUM / HIGH / CRITICAL)
│   ├── logger.py               # Standardized application logging
│   └── routers/
│       ├── auth.py             # POST /api/v1/auth/register, login, me
│       ├── fraud.py            # POST /api/v1/fraud/predict, history
│       ├── models.py           # POST/GET/PATCH /api/v1/models (Model Registry & A/B weights)
│       ├── metrics.py          # GET /api/v1/metrics (System & model evaluation KPI telemetry)
│       └── sample.py           # GET /api/v1/sample-csv (Synthetic transaction generator)
├── frontend/                   # React 18 + Vite + TypeScript SPA
│   ├── src/
│   │   ├── components/         # Clean UI components (Navbar, Login, PredictTab, MetricsTab, ModelsTab)
│   │   ├── api.ts              # Strongly-typed fetch API wrapper
│   │   ├── index.css           # Modern glassmorphism dark-theme design system
│   │   └── main.tsx            # React root mount
│   └── dist/                   # Built production static assets served directly by FastAPI
├── ml/                         # Machine Learning Pipeline
│   ├── preprocessing/          # Data transformations & scaling
│   ├── training/               # XGBoost model training & evaluation scripts
│   ├── train_models.py         # Multi-version training script (v1-champion, v2-high-recall, v3-high-precision)
│   └── utils.py                # Joblib artifact serialization utilities
├── saved_models/               # Serialized model (.joblib) & metadata (.json) artifacts
├── tests/                      # Comprehensive pytest test suite (29/29 passing)
├── Dockerfile                  # Multi-stage Docker build (Node frontend build -> Python runtime)
├── docker-compose.yml          # Full-stack container orchestration (PostgreSQL + API + UI)
├── entrypoint.sh               # Startup script for automatic Alembic migrations
├── render.yaml                 # Render cloud deployment blueprint
└── requirements.txt            # Python backend dependencies
```

---

## Key Features

1. **Cohesive Single-Page Application**: A modern React 18 + Vite + TypeScript frontend with a sleek glassmorphic dark theme, responsive data visualizations, and interactive A/B testing sliders. Served directly by FastAPI in production.
2. **XGBoost Inference Engine**: Evaluates transactions in real time (<100ms) against trained gradient boosted decision trees with custom log1p and StandardScaler preprocessing pipelines.
3. **Dynamic Model Registry & Weighted A/B Testing**:
   - **v1-champion**: Standard balanced threshold (0.90).
   - **v2-high-recall**: High sensitivity (0.65 threshold) designed to catch subtle fraud patterns.
   - **v3-high-precision**: High specificity (0.95 threshold) minimizing false positives.
   - Dynamically route production inference traffic across active models using customizable percentage weights.
4. **Real-Time Telemetry & Analytics**: Filter precision, recall, F1, accuracy, ROC AUC, uptime, and risk distribution by specific model version tags directly from the UI dashboard.
5. **Synthetic Sample Data Generator**: Instantly download randomly generated transaction CSVs with legitimate features and realistic fraud signatures to test inference on demand.

---

## Default Credentials

When booting for the first time, the system automatically seeds a default administrator account in SQLite / PostgreSQL:
- **Username**: `admin`
- **Password**: `admin123`

---

## Quick Start

### Docker Compose (Recommended)

Spin up the full stack (PostgreSQL database, FastAPI backend with automatic Alembic migrations, and built React UI) with a single command:

```bash
docker compose up --build
```
Visit **http://localhost:8000** in your browser to access the dashboard. API documentation is available at **http://localhost:8000/docs**.

---

### Local Development

1. **Install Backend Dependencies & Train Models**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   python ml/train_models.py
   ```

2. **Build React Frontend**:
   ```bash
   cd frontend
   npm install
   npm run build
   cd ..
   ```

3. **Start the Unified Server**:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```
   Visit **http://localhost:8000** in your browser to access the dashboard. API documentation is available at **http://localhost:8000/docs**.

### Running Tests

Run the full automated pytest suite:
```bash
pytest -v
```

---

## Cloud Deployment (Render / Docker)

This repository includes a 100% cloud-ready `render.yaml` blueprint and a multi-stage `Dockerfile` that compiles the React frontend and packages the FastAPI runtime into a single container.

1. Connect this repository to [Render.com](https://render.com).
2. Create a new **Blueprint** and select `render.yaml`.
3. The platform will provision a managed PostgreSQL database and build the Docker container automatically.
