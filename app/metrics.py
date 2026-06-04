"""
GET /stores/{store_id}/metrics — real-time store metrics for today.
GET /stores/{store_id}/heatmap — zone visit frequency & dwell heatmap.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone

import structlog
from fastapi import APIRouter, Depends, Path, Query
from fastapi.responses import JSONResponse
from sqlalchemy import and_, case, distinct, func, select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import (
    EventRecord,
    HeatmapCell,
    MetricsResponse,
    PosTransaction,
)

log = structlog.get_logger()
router = APIRouter(tags=["metrics"])


@router.get("/stores", tags=["metrics"])
def get_stores(db: Session = Depends(get_db)) -> list[str]:
    """Return a list of all distinct store_ids that have events."""
    store_ids = db.scalars(select(distinct(EventRecord.store_id))).all()
    # Filter out nulls
    return sorted(s for s in store_ids if s)


@router.get(
    "/stores/{store_id}/metrics",
    response_model=MetricsResponse,
    responses={503: {"description": "Database unavailable"}},
)
def get_store_metrics(
    store_id: str = Path(..., description="Store identifier"),
    date: str = Query(None, description="Date in YYYY-MM-DD format"),
    db: Session = Depends(get_db),
) -> MetricsResponse | JSONResponse:
    try:
        # ---- unique_visitors (non-staff) ----
        unique_visitors: int = (
            db.scalar(
                select(func.count(distinct(EventRecord.visitor_id))).where(
                    and_(
                        EventRecord.store_id == store_id,
                        EventRecord.is_staff == False,  # noqa: E712
                    )
                )
            )
            or 0
        )

        # ---- conversion_rate ----
        # Matches billing queue joins to POS transactions within a 5-minute
        # window using SQLAlchemy ORM — works on both SQLite (tests) and
        # PostgreSQL (production).
        converted = 0
        if unique_visitors > 0:
            window = timedelta(minutes=5)

            billing_rows = db.execute(
                select(
                    EventRecord.visitor_id,
                    func.min(EventRecord.timestamp).label("billing_ts"),
                ).where(
                    and_(
                        EventRecord.store_id == store_id,
                        EventRecord.is_staff == False,  # noqa: E712
                        EventRecord.event_type.in_(
                            ["BILLING_QUEUE_JOIN", "ZONE_ENTER"]
                        ),
                        EventRecord.zone_id.contains("billing"),
                    )
                ).group_by(EventRecord.visitor_id)
            ).all()

            pos_rows = db.execute(
                select(PosTransaction.timestamp).where(
                    PosTransaction.store_id == store_id,
                )
            ).scalars().all()

            # Sort POS timestamps once so we can break early per billing row.
            # Normalise all to naive UTC to avoid tz-aware vs tz-naive TypeError
            # (SQLite returns naive datetimes even for DateTime(timezone=True) columns).
            def _to_naive(dt: datetime) -> datetime:
                if dt is None:
                    return datetime.min
                return dt.replace(tzinfo=None) if dt.tzinfo is not None else dt

            pos_sorted = sorted(_to_naive(ts) for ts in pos_rows)
            converted_visitors: set[str] = set()
            for brow in billing_rows:
                bts = _to_naive(brow.billing_ts)
                for pos_ts in pos_sorted:
                    if pos_ts < bts:
                        continue
                    if pos_ts > bts + window:
                        break
                    converted_visitors.add(brow.visitor_id)
                    break
            converted = len(converted_visitors)

        conversion_rate = round(converted / unique_visitors, 4) if unique_visitors else 0.0

        # ---- avg_dwell_per_zone ----
        dwell_rows = db.execute(
            select(
                EventRecord.zone_id,
                func.avg(EventRecord.dwell_ms).label("avg_dwell"),
            )
            .where(
                and_(
                    EventRecord.store_id == store_id,
                    EventRecord.event_type == "ZONE_DWELL",
                    EventRecord.is_staff == False,  # noqa: E712
                    EventRecord.zone_id.isnot(None),
                )
            )
            .group_by(EventRecord.zone_id)
        ).all()

        avg_dwell_per_zone = {
            row.zone_id: round(float(row.avg_dwell), 2) for row in dwell_rows
        }

        # ---- current_queue_depth ----
        joined_billing = (
            select(distinct(EventRecord.visitor_id).label("visitor_id")).where(
                and_(
                    EventRecord.store_id == store_id,
                    EventRecord.is_staff == False,  # noqa: E712
                    EventRecord.event_type == "BILLING_QUEUE_JOIN",
                )
            )
        ).subquery("joined")

        exited_billing = (
            select(distinct(EventRecord.visitor_id).label("visitor_id")).where(
                and_(
                    EventRecord.store_id == store_id,
                    EventRecord.is_staff == False,  # noqa: E712
                    EventRecord.event_type.in_(["ZONE_EXIT", "BILLING_QUEUE_ABANDON"]),
                    EventRecord.zone_id.contains("billing"),
                )
            )
        ).subquery("exited")

        current_queue_depth: int = (
            db.scalar(
                select(func.count()).select_from(
                    select(joined_billing.c.visitor_id)
                    .where(joined_billing.c.visitor_id.notin_(select(exited_billing.c.visitor_id)))
                    .subquery()
                )
            )
            or 0
        )

        # ---- abandonment_rate ----
        joins_count: int = (
            db.scalar(
                select(func.count()).where(
                    and_(
                        EventRecord.store_id == store_id,
                        EventRecord.is_staff == False,  # noqa: E712
                        EventRecord.event_type == "BILLING_QUEUE_JOIN",
                    )
                )
            )
            or 0
        )

        abandon_count: int = (
            db.scalar(
                select(func.count()).where(
                    and_(
                        EventRecord.store_id == store_id,
                        EventRecord.is_staff == False,  # noqa: E712
                        EventRecord.event_type == "BILLING_QUEUE_ABANDON",
                    )
                )
            )
            or 0
        )

        abandonment_rate = round(abandon_count / joins_count, 4) if joins_count else 0.0

        return MetricsResponse(
            store_id=store_id,
            date="ALL_TIME",
            unique_visitors=unique_visitors,
            conversion_rate=conversion_rate,
            avg_dwell_per_zone=avg_dwell_per_zone,
            current_queue_depth=current_queue_depth,
            abandonment_rate=abandonment_rate,
        )

    except Exception as exc:
        log.error("metrics_error", store_id=store_id, error=str(exc))
        return JSONResponse(
            status_code=503,
            content={"error": "service_unavailable", "detail": str(exc)},
        )


@router.get(
    "/stores/{store_id}/heatmap",
    response_model=list[HeatmapCell],
    responses={503: {"description": "Database unavailable"}},
)
def get_store_heatmap(
    store_id: str = Path(..., description="Store identifier"),
    date: str = Query(None, description="Date in YYYY-MM-DD format"),
    db: Session = Depends(get_db),
) -> list[HeatmapCell] | JSONResponse:
    try:
        total_sessions: int = (
            db.scalar(
                select(func.count(distinct(EventRecord.visitor_id))).where(
                    and_(
                        EventRecord.store_id == store_id,
                        EventRecord.is_staff == False,  # noqa: E712
                    )
                )
            )
            or 0
        )
        confidence = "high" if total_sessions >= 20 else "low"

        zone_visits = db.execute(
            select(
                EventRecord.zone_id,
                func.count().label("visit_count"),
            )
            .where(
                and_(
                    EventRecord.store_id == store_id,
                    EventRecord.is_staff == False,  # noqa: E712
                    EventRecord.event_type.in_(["ZONE_ENTER", "ZONE_DWELL"]),
                    EventRecord.zone_id.isnot(None),
                )
            )
            .group_by(EventRecord.zone_id)
        ).all()

        zone_dwells = db.execute(
            select(
                EventRecord.zone_id,
                func.avg(EventRecord.dwell_ms).label("avg_dwell"),
            )
            .where(
                and_(
                    EventRecord.store_id == store_id,
                    EventRecord.event_type == "ZONE_DWELL",
                    EventRecord.is_staff == False,  # noqa: E712
                    EventRecord.zone_id.isnot(None),
                )
            )
            .group_by(EventRecord.zone_id)
        ).all()

        dwell_map = {r.zone_id: float(r.avg_dwell) for r in zone_dwells}

        if not zone_visits:
            return []

        max_visits = max(r.visit_count for r in zone_visits)

        cells: list[HeatmapCell] = []
        for row in zone_visits:
            normalized = round((row.visit_count / max_visits) * 100, 2) if max_visits else 0
            cells.append(
                HeatmapCell(
                    zone_id=row.zone_id,
                    visit_count=row.visit_count,
                    avg_dwell_ms=round(dwell_map.get(row.zone_id, 0.0), 2),
                    normalized_score=normalized,
                    data_confidence=confidence,
                )
            )

        return cells

    except Exception as exc:
        log.error("heatmap_error", store_id=store_id, error=str(exc))
        return JSONResponse(
            status_code=503,
            content={"error": "service_unavailable", "detail": str(exc)},
        )
