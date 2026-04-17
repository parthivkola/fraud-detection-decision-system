# How to Run — Fraud Detection Decision System

## Option A: Local Development (Recommended for First Run)

### Prerequisites
- Python 3.10+
- PostgreSQL 14+ running locally
- Git

### Step-by-Step

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/fraud-detection-decision-system.git
cd fraud-detection-decision-system

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create your .env file
cat > .env << 'EOF'
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/fraud_detection
SECRET_KEY=your-super-secret-key-change-this
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
DEFAULT_ADMIN_USERNAME=admin
DEFAULT_ADMIN_EMAIL=admin@fraudapi.local
DEFAULT_ADMIN_PASSWORD=admin123
DEBUG=true
EOF

# 5. Create the database in PostgreSQL
psql -U postgres -c "CREATE DATABASE fraud_detection;"

# 6. Run database migrations
alembic upgrade head

# 7. Start the server
uvicorn app.main:app --reload
```

### What Happens on First Startup
1. Tables are created if they don't exist
2. Default admin user (`admin / admin123`) is seeded
3. XGBoost model + scaler + metadata are loaded from `saved_models/`
4. Server starts on `http://localhost:8000`

### Access Points
| URL | What |
|-----|------|
| `http://localhost:8000/dashboard` | Frontend UI |
| `http://localhost:8000/docs` | Swagger API docs |
| `http://localhost:8000/health` | Health check |

### Quick Test
```bash
# Login and get a token
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","email":"admin@fraudapi.local","password":"admin123"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

echo "Token: $TOKEN"

# Check metrics
curl -s http://localhost:8000/api/v1/metrics \
  -H "Authorization: Bearer $TOKEN" | python3 -m json.tool
```

---

## Option B: Docker (One Command)

```bash
# Just run this — PostgreSQL + API, everything included
docker compose up --build -d

# Check it's running
docker compose ps
docker compose logs api --tail 20

# Access
open http://localhost:8000/dashboard
```

### Stop & Clean Up
```bash
docker compose down            # Stop containers
docker compose down -v         # Stop + delete database volume
```

---

## Running Tests

```bash
source venv/bin/activate

# Run all 29 tests
pytest tests/ -v

# Run a specific file
pytest tests/test_auth.py -v

# Run with coverage report
pytest tests/ --cov=app --cov-report=term-missing
```

Tests use an in-memory SQLite database — no PostgreSQL needed.

---

## Common Issues

| Problem | Fix |
|---------|-----|
| `connection refused` to PostgreSQL | Make sure PostgreSQL is running: `sudo systemctl start postgresql` |
| `database "fraud_detection" does not exist` | Run: `psql -U postgres -c "CREATE DATABASE fraud_detection;"` |
| `bcrypt` error with passlib | Run: `pip install bcrypt==4.2.1` (version 5.x has compatibility issues) |
| Model files not found | Make sure `saved_models/` contains `xgb_model.joblib`, `amount_scaler.joblib`, `model_metadata.json` |
| Port 8000 already in use | Kill the old process: `lsof -ti:8000 | xargs kill -9` |
