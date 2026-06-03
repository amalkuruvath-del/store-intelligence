"""
POST /events/ingest — batch event ingestion.

* Accepts up to 500 events per call.
* Idempotent by event_id (duplicates skipped silently).
* Partial success: valid events ingested, malformed ones reported in errors[].
"""

from __future__ import annotations

from typing import Any, List

import structlog
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import EventIn, EventRecord, IngestError, IngestResponse

log = structlog.get_logger()

router = APIRouter(tags=["ingestion"])


@router.post(
    "/events/ingest",
    response_model=IngestResponse,
    status_code=200,
    responses={503: {"description": "Database unavailable"}},
)
def ingest_events(
    request: Request,
    payload: List[Any],
    db: Session = Depends(get_db),
) -> IngestResponse | JSONResponse:
    """
    Ingest a batch of camera/sensor events.

    * Maximum 500 events per request.
    * Duplicate event_ids are silently skipped.
    * Malformed events are collected into ``errors``.
    """
    errors: list[IngestError] = []
    accepted = 0

    if len(payload) > 500:
        return JSONResponse(
            status_code=400,
            content={
                "error": "batch_too_large",
                "detail": f"Maximum 500 events per request, received {len(payload)}.",
            },
        )

    try:
        # Gather all event_ids from the batch to check for existing duplicates
        candidate_ids: list[str] = []
        valid_events: list[tuple[int, EventIn]] = []

        for idx, raw in enumerate(payload):
            try:
                ev = EventIn.model_validate(raw)
                valid_events.append((idx, ev))
                candidate_ids.append(ev.event_id)
            except (ValidationError, Exception) as exc:
                errors.append(IngestError(index=idx, reason=str(exc)))

        # Query existing event_ids in one round-trip
        existing_ids: set[str] = set()
        if candidate_ids:
            rows = db.execute(
                select(EventRecord.event_id).where(
                    EventRecord.event_id.in_(candidate_ids)
                )
            ).scalars().all()
            existing_ids = set(rows)

        # Build ORM objects for new events only
        to_insert: list[EventRecord] = []
        for idx, ev in valid_events:
            if ev.event_id in existing_ids:
                # Duplicate — skip silently (idempotent)
                continue
            to_insert.append(
                EventRecord(
                    event_id=ev.event_id,
                    store_id=ev.store_id,
                    camera_id=ev.camera_id,
                    visitor_id=ev.visitor_id,
                    event_type=ev.event_type,
                    timestamp=ev.timestamp,
                    zone_id=ev.zone_id,
                    dwell_ms=ev.dwell_ms,
                    is_staff=ev.is_staff,
                    confidence=ev.confidence,
                    meta_queue_depth=ev.metadata.queue_depth,
                    meta_sku_zone=ev.metadata.sku_zone,
                    meta_session_seq=ev.metadata.session_seq,
                )
            )

        if to_insert:
            db.add_all(to_insert)
            db.commit()

        accepted = len(to_insert)
        rejected = len(errors)

        # Log event_count for the middleware enrichment
        log.info(
            "ingest_complete",
            accepted=accepted,
            rejected=rejected,
            duplicates_skipped=len(valid_events) - accepted,
            trace_id=getattr(request.state, "trace_id", None),
        )

        return IngestResponse(accepted=accepted, rejected=rejected, errors=errors)

    except Exception as exc:
        db.rollback()
        log.error("ingest_db_error", error=str(exc))
        return JSONResponse(
            status_code=503,
            content={"error": "service_unavailable", "detail": str(exc)},
        )
