"""
GET /stores/{store_id}/funnel — conversion funnel.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone

import structlog
from fastapi import APIRouter, Depends, Path, Query
from fastapi.responses import JSONResponse
from sqlalchemy import and_, distinct, func, select
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import (
    EventRecord,
    FunnelResponse,
    FunnelStage,
    PosTransaction,
)

log = structlog.get_logger()
router = APIRouter(tags=["funnel"])

@router.get(
    "/stores/{store_id}/funnel",
    response_model=FunnelResponse,
    responses={503: {"description": "Database unavailable"}},
)
def get_funnel(
    store_id: str = Path(..., description="Store identifier"),
    date: str = Query(None, description="Date in YYYY-MM-DD format"),
    db: Session = Depends(get_db),
) -> FunnelResponse | JSONResponse:
    try:
        base_filter = and_(
            EventRecord.store_id == store_id,
            EventRecord.is_staff == False,  # noqa: E712
        )

        # Stage 1: Entry — unique visitors with ENTRY or REENTRY
        entry_count: int = (
            db.scalar(
                select(func.count(distinct(EventRecord.visitor_id))).where(
                    and_(
                        base_filter,
                        EventRecord.event_type.in_(["ENTRY", "REENTRY"]),
                    )
                )
            )
            or 0
        )

        # Stage 2: Zone Visit — unique visitors with at least one ZONE_ENTER
        zone_visit_count: int = (
            db.scalar(
                select(func.count(distinct(EventRecord.visitor_id))).where(
                    and_(
                        base_filter,
                        EventRecord.event_type == "ZONE_ENTER",
                    )
                )
            )
            or 0
        )

        # Stage 3: Billing Queue — unique visitors with BILLING_QUEUE_JOIN
        billing_queue_count: int = (
            db.scalar(
                select(func.count(distinct(EventRecord.visitor_id))).where(
                    and_(
                        base_filter,
                        EventRecord.event_type == "BILLING_QUEUE_JOIN",
                    )
                )
            )
            or 0
        )

        # Stage 4: Purchase — billing queue visitors correlated with POS txn
        purchase_count = 0
        if billing_queue_count > 0:
            billing_rows = db.execute(
                select(
                    EventRecord.visitor_id,
                    func.min(EventRecord.timestamp).label("join_ts"),
                )
                .where(
                    and_(
                        base_filter,
                        EventRecord.event_type == "BILLING_QUEUE_JOIN",
                    )
                )
                .group_by(EventRecord.visitor_id)
            ).all()

            pos_rows = db.execute(
                select(PosTransaction.timestamp).where(
                    and_(
                        PosTransaction.store_id == store_id,
                    )
                )
            ).scalars().all()

            converted_visitors = set()
            for brow in billing_rows:
                for pos_ts in pos_rows:
                    if pos_ts >= brow.join_ts and pos_ts <= brow.join_ts + timedelta(minutes=5):
                        converted_visitors.add(brow.visitor_id)
                        break

            purchase_count = len(converted_visitors)

        def _drop(prev: int, cur: int) -> float:
            if prev == 0:
                return 0.0
            return round(((prev - cur) / prev) * 100, 2)

        stages = [
            FunnelStage(stage="Entry", count=entry_count, drop_off_pct=0.0),
            FunnelStage(
                stage="Zone Visit",
                count=zone_visit_count,
                drop_off_pct=_drop(entry_count, zone_visit_count),
            ),
            FunnelStage(
                stage="Billing Queue",
                count=billing_queue_count,
                drop_off_pct=_drop(zone_visit_count, billing_queue_count),
            ),
            FunnelStage(
                stage="Purchase",
                count=purchase_count,
                drop_off_pct=_drop(billing_queue_count, purchase_count),
            ),
        ]

        return FunnelResponse(store_id=store_id, stages=stages)

    except Exception as exc:
        log.error("funnel_error", store_id=store_id, error=str(exc))
        return JSONResponse(
            status_code=503,
            content={"error": "service_unavailable", "detail": str(exc)},
        )
