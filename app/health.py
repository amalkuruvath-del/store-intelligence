"""
GET /health — service health check.

Reports:
  * DB connectivity
  * Per-store last event timestamp
  * Stale feeds (last event > 10 min ago)
  * Process uptime
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import structlog
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import distinct, func, select, text
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import EventRecord, HealthResponse

log = structlog.get_logger()
router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    responses={503: {"description": "Database unavailable"}},
)
def health_check(
    db: Session = Depends(get_db),
) -> HealthResponse | JSONResponse:
    # Import here to avoid circular import at module load time
    from app.main import get_startup_ts

    db_connected = False
    last_event_per_store: dict[str, str | None] = {}
    stale_feeds: list[str] = []

    try:
        # Verify DB connectivity
        db.execute(text("SELECT 1"))
        db_connected = True

        # Last event timestamp per store
        rows = db.execute(
            select(
                EventRecord.store_id,
                func.max(EventRecord.timestamp).label("last_ts"),
            ).group_by(EventRecord.store_id)
        ).all()

        now = datetime.now(timezone.utc)
        for row in rows:
            ts: datetime | None = row.last_ts
            if ts is not None:
                last_event_per_store[row.store_id] = ts.isoformat()
                if (now - ts) > timedelta(minutes=10):
                    stale_feeds.append(row.store_id)
            else:
                last_event_per_store[row.store_id] = None

    except Exception as exc:
        log.error("health_db_error", error=str(exc))
        # If DB fails, we still return a degraded response instead of 503
        # so monitoring tools can see the service is running.

    status = "ok" if db_connected else "degraded"
    startup_ts = get_startup_ts()
    uptime = time.time() - startup_ts if startup_ts > 0 else 0.0

    return HealthResponse(
        status=status,
        db_connected=db_connected,
        last_event_per_store=last_event_per_store,
        stale_feeds=stale_feeds,
        uptime_seconds=round(uptime, 2),
    )
