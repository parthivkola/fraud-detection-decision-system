FROM python:3.11-slim

WORKDIR /app

# System deps for psycopg2 and bcrypt
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY ml/ ./ml/
COPY saved_models/ ./saved_models/
COPY alembic/ ./alembic/
COPY alembic.ini .

# Create dirs that the app expects
RUN mkdir -p logs data/raw data/processed

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
