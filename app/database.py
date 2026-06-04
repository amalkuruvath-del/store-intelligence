"""
Database engine, session factory, and Base class for SQLAlchemy 2.0.

Reads DATABASE_URL from environment (default: PostgreSQL on Docker Compose).
Provides a `get_db` FastAPI dependency that yields a session per request.

IMPORTANT: The engine is created lazily on first use so that importing this
module during testing (where conftest.py overrides get_db with SQLite) does
NOT immediately attempt a TCP connection to the PostgreSQL host.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from typing import Any

from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@db:5432/store_intelligence",
)

# ── Lazy engine ───────────────────────────────────────────────────────────────
# _engine is None until the first call to _get_engine().
# This prevents a TCP connection attempt at import time, which would crash
# pytest when running locally without Docker (host "db" doesn't resolve).
_engine = None
_SessionLocal = None


def _get_engine():
    """Return the singleton SQLAlchemy engine, creating it on first call."""
    global _engine, _SessionLocal
    if _engine is None:
        kwargs: dict[str, Any] = {}
        if DATABASE_URL.startswith("postgresql"):
            kwargs = {
                "pool_size": 10,
                "max_overflow": 20,
                "pool_pre_ping": True,
            }
        _engine = create_engine(DATABASE_URL, echo=False, **kwargs)
        _SessionLocal = sessionmaker(
            bind=_engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )
    return _engine, _SessionLocal


class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""
    pass


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency — yields a DB session and closes it after the request."""
    _, SessionLocal = _get_engine()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
