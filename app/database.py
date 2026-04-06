"""
Database connection setup.

Three key pieces:
  1. engine         -the actual connection to PostgreSQL.
                     SQLAlchemy uses this to send SQL statements.
  2. SessionLocal   -a *factory* that creates new database sessions.
                     Each API request gets its own session so that
                     one request's work doesn't bleed into another.
  3. get_db()       -a FastAPI *dependency*.  It yields a session,
                     and guarantees the session is closed when the
                     request finishes (even if it crashes).

How it's used:
    In any route, you write:
        def my_route(db: Session = Depends(get_db)):
    FastAPI then automatically calls get_db(), gives you a session,
    and cleans it up after the response is sent.
"""

from __future__ import annotations

from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from app.config import settings

# ---- 1. Engine ----
# create_engine() doesn't connect immediately; it creates a *pool*
# of connections that are reused across requests.
engine = create_engine(
    settings.DATABASE_URL,
    # pool_pre_ping=True makes SQLAlchemy test each connection before
    # using it, so stale/dropped connections are automatically replaced.
    pool_pre_ping=True,
)

# ---- 2. Session factory ----
# sessionmaker() returns a *class* (not an instance).
# Every time you call SessionLocal(), you get a brand-new session.
#
# autocommit=False  → you must explicitly call db.commit()
# autoflush=False   → SQLAlchemy won't auto-send pending changes
#                      to the DB before every query.  This gives you
#                      more control over when writes happen.
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
)

# ---- 3. Base class for models ----
# All your ORM models (in models.py) will inherit from this Base.
# It keeps a registry of all models, which is used by
# Base.metadata.create_all() to auto-create tables.
Base = declarative_base()


# ---- 4. Dependency for FastAPI ----
def get_db() -> Generator[Session, None, None]:
    """
    Yield a database session for one request, then close it.

    Usage in a route:
        @router.get("/stuff")
        def get_stuff(db: Session = Depends(get_db)):
            return db.query(MyModel).all()

    The ``yield`` turns this into a context-manager-like dependency:
      - Everything before yield runs BEFORE the route handler
      - Everything after yield runs AFTER the response is sent
      - The finally block guarantees cleanup even on errors
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
