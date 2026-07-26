# ── Stage 1: Build React Frontend ──────────────────────────────────────────────
FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci || npm install
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python Backend Runtime ───────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# System deps for psycopg2 and bcrypt
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer-cached)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY ml/ ./ml/
COPY saved_models/ ./saved_models/
COPY alembic/ ./alembic/
COPY alembic.ini .

# Copy built frontend assets from stage 1
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# Runtime directories — set permissions before the volume can shadow them
RUN mkdir -p logs data/raw data/processed && chmod -R 777 logs data

EXPOSE 8000

# entrypoint.sh runs DB migrations then starts the server
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
