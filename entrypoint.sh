#!/usr/bin/env sh
set -e

echo "==> Checking migration state..."

# Check if alembic_version table exists (i.e., DB has been migrated before)
ALEMBIC_TABLE_EXISTS=$(python3 - <<'EOF'
import os, psycopg2, sys
try:
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    cur = conn.cursor()
    cur.execute("SELECT to_regclass('public.alembic_version')")
    result = cur.fetchone()[0]
    conn.close()
    print("yes" if result else "no")
except Exception as e:
    print("no")
EOF
)

if [ "$ALEMBIC_TABLE_EXISTS" = "no" ]; then
    echo "==> Fresh database detected — letting SQLAlchemy create all tables, then stamping alembic head."
    # Tables are created by app startup (Base.metadata.create_all).
    # We stamp after so alembic knows schema is at current revision.
    python3 -c "
from app.database import engine
from app.models import Base
Base.metadata.create_all(bind=engine)
print('Tables created.')
"
    alembic stamp head
    echo "==> Alembic stamped at head."
else
    echo "==> Existing database — running any pending migrations."
    alembic upgrade head
    echo "==> Migrations complete."
fi

echo "==> Starting API server..."
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
