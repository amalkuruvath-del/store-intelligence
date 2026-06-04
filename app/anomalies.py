"""
GET /stores/{store_id}/anomalies — active store anomalies.

Detects three rule-based anomalies from the events database:
  1. Long billing queue  — queue depth > 3 sustained for > 5 minutes  (WARN)
  2. Dead store          — zero customer activity for > 30 minutes     (INFO)
  3. Excessive dwell     — any customer in a zone > 45 minutes         (WARN)

Uses SQLAlchemy ORM exclusively (no raw SQL) so queries run on both
SQLite (test suite) and PostgreSQL (production) without modification.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import structlog
from fastapi import APIRouter, Depends, Path, Query
from fastapi.responses import JSONResponse
from sqlalchemy import and_, distinct, func, not_, select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import AnomalyResponse, AnomalyItem, EventRecord

log = structlog.get_logger()
router = APIRouter(tags=["anomalies"])

# ── Thresholds ─────────────────────────────────────────────────────────────
_QUEUE_DEPTH_THRESHOLD = 3
_QUEUE_SUSTAINED_MINUTES = 5
_DEAD_STORE_MINUTES = 30
_EXCESSIVE_DWELL_MINUTES = 45


@router.get(
    "/stores/{store_id}/anomalies",
    response_model=AnomalyResponse,
    responses={503: {"description": "Database unavailable"}},
)
def get_store_anomalies(
    store_id: str = Path(..., description="Store identifier"),
    date: str = Query(None, description="Date in YYYY-MM-DD format"),
    db: Session = Depends(get_db),
) -> AnomalyResponse | JSONResponse:
    """Evaluate active store rules and return a list of triggered anomalies."""
    try:
        now = datetime.now(timezone.utc)
        anomalies: list[AnomalyItem] = []

        # ── 1. Long billing queue ─────────────────────────────────────────
        # Count visitors who joined the billing queue more than
        # _QUEUE_SUSTAINED_MINUTES ago and have NOT left yet.
        sustained_cutoff = now - timedelta(minutes=_QUEUE_SUSTAINED_MINUTES)

        # Visitors who joined billing queue before the cutoff
        joined_sq = (
            select(
                EventRecord.visitor_id,
                func.min(EventRecord.timestamp).label("join_ts"),
            )
            .where(
                and_(
                    EventRecord.store_id == store_id,
                    EventRecord.is_staff == False,  # noqa: E712
                    EventRecord.event_type == "BILLING_QUEUE_JOIN",
                )
            )
            .group_by(EventRecord.visitor_id)
            .subquery("joined")
        )

        # Visitors who have since left the billing zone
        left_sq = (
            select(distinct(EventRecord.visitor_id).label("visitor_id"))
            .where(
                and_(
                    EventRecord.store_id == store_id,
                    EventRecord.event_type.in_(
                        ["BILLING_QUEUE_ABANDON", "ZONE_EXIT"]
                    ),
                    EventRecord.zone_id.contains("billing"),
                )
            )
            .subquery("left_billing")
        )

        still_waiting = (
            db.scalar(
                select(func.count())
                .select_from(joined_sq)
                .where(
                    and_(
                        joined_sq.c.visitor_id.notin_(
                            select(left_sq.c.visitor_id)
                        ),
                        joined_sq.c.join_ts <= sustained_cutoff,
                    )
                )
            )
            or 0
        )

        if still_waiting > _QUEUE_DEPTH_THRESHOLD:
            anomalies.append(
                AnomalyItem(
                    type="long_billing_queue",
                    severity="WARN",
                    zone_id="billing",
                    detail=(
                        f"{still_waiting} customers have been waiting in the "
                        f"billing queue for over {_QUEUE_SUSTAINED_MINUTES} minutes."
                    ),
                    suggested_action="Open an additional billing counter immediately.",
                    detected_at=now,
                )
            )

        # ── 2. Dead store — no customer events for > 30 minutes ───────────
        dead_cutoff = now - timedelta(minutes=_DEAD_STORE_MINUTES)

        last_event_ts = db.scalar(
            select(func.max(EventRecord.timestamp)).where(
                and_(
                    EventRecord.store_id == store_id,
                    EventRecord.is_staff == False,  # noqa: E712
                )
            )
        )

        if last_event_ts is not None:
            if last_event_ts.tzinfo is None:
                last_event_ts = last_event_ts.replace(tzinfo=timezone.utc)
            if last_event_ts < dead_cutoff:
                idle_minutes = int((now - last_event_ts).total_seconds() / 60)
                anomalies.append(
                    AnomalyItem(
                        type="no_customer_activity",
                        severity="INFO",
                        zone_id=None,
                        detail=(
                            f"No customer activity detected for {idle_minutes} minutes. "
                            f"Last event at {last_event_ts.strftime('%H:%M:%S')} UTC."
                        ),
                        suggested_action=(
                            "Verify camera feeds are live and the store is open."
                        ),
                        detected_at=now,
                    )
                )

        # ── 3. Excessive zone dwell — visitor in a zone > 45 minutes ──────
        dwell_cutoff = now - timedelta(minutes=_EXCESSIVE_DWELL_MINUTES)

        # Latest ZONE_ENTER per visitor+zone
        enter_sq = (
            select(
                EventRecord.visitor_id,
                EventRecord.zone_id,
                func.max(EventRecord.timestamp).label("enter_ts"),
            )
            .where(
                and_(
                    EventRecord.store_id == store_id,
                    EventRecord.is_staff == False,  # noqa: E712
                    EventRecord.event_type == "ZONE_ENTER",
                    EventRecord.zone_id.isnot(None),
                )
            )
            .group_by(EventRecord.visitor_id, EventRecord.zone_id)
            .subquery("enters")
        )

        # Visitor+zone combos that have a ZONE_EXIT
        exited_sq = (
            select(
                EventRecord.visitor_id,
                EventRecord.zone_id,
            )
            .where(
                and_(
                    EventRecord.store_id == store_id,
                    EventRecord.event_type == "ZONE_EXIT",
                    EventRecord.zone_id.isnot(None),
                )
            )
            .subquery("exits")
        )

        # Still-in-zone visitors whose entry was before the dwell cutoff
        long_dwellers = db.execute(
            select(
                enter_sq.c.visitor_id,
                enter_sq.c.zone_id,
                enter_sq.c.enter_ts,
            ).where(
                and_(
                    enter_sq.c.enter_ts <= dwell_cutoff,
                    not_(
                        select(func.count())
                        .where(
                            and_(
                                exited_sq.c.visitor_id == enter_sq.c.visitor_id,
                                exited_sq.c.zone_id == enter_sq.c.zone_id,
                            )
                        )
                        .correlate(enter_sq)
                        .scalar_subquery()
                        > 0
                    ),
                )
            )
        ).all()

        for row in long_dwellers:
            enter_ts = row.enter_ts
            if enter_ts.tzinfo is None:
                enter_ts = enter_ts.replace(tzinfo=timezone.utc)
            dwell_minutes = int((now - enter_ts).total_seconds() / 60)
            anomalies.append(
                AnomalyItem(
                    type="excessive_zone_dwell",
                    severity="WARN",
                    zone_id=row.zone_id,
                    detail=(
                        f"Visitor {row.visitor_id} has been in zone "
                        f"'{row.zone_id}' for {dwell_minutes} minutes."
                    ),
                    suggested_action=(
                        "Send staff to assist the customer or verify zone "
                        "exit events are being captured correctly."
                    ),
                    detected_at=now,
                )
            )

        return AnomalyResponse(store_id=store_id, anomalies=anomalies)

    except Exception as exc:
        log.error("anomalies_error", store_id=store_id, error=str(exc))
        return JSONResponse(
            status_code=503,
            content={"error": "service_unavailable", "detail": str(exc)},
        )
