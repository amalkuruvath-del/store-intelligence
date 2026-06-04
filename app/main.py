"""
FastAPI application entry-point.

* Lifespan handler — creates tables and seeds POS data on startup.
* Structured JSON logging middleware via structlog.
* Graceful-degradation error handling (503 on DB failures).
"""

from __future__ import annotations

import csv
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, AsyncGenerator

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from sqlalchemy import select, text

from app.database import Base, _get_engine
from app.models import PosTransaction

# ---------------------------------------------------------------------------
# structlog configuration — JSON to stdout
# ---------------------------------------------------------------------------
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(0),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Track startup time for /health uptime calculation
# ---------------------------------------------------------------------------
_STARTUP_TS: float = 0.0


def _seed_pos_data() -> None:
    """Load /data/pos_transactions.csv into PosTransaction table if it is empty."""
    _, SessionLocal = _get_engine()
    db = SessionLocal()
    try:
        count = db.scalar(select(PosTransaction.id).limit(1))
        if count is not None:
            log.info("pos_seed_skip", reason="table_not_empty")
            return

        csv_path = Path("/data/pos_transactions.csv")
        if not csv_path.exists():
            log.info("pos_seed_skip", reason="csv_not_found", path=str(csv_path))
            return

        records: list[PosTransaction] = []
        with csv_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)  # skip header row
            for row in reader:
                # order_id,order_date,order_time,store_id,product_id,brand_name,total_amount
                if len(row) < 7:
                    continue
                transaction_id = row[0].strip()
                time_str = row[2].strip()
                # We map ST1008 to STORE_01 since ST1008 is just dummy data
                store_id = row[3].strip()
                if store_id == "ST1008":
                    store_id = "STORE_01"  # Force map for demonstration
                value_str = row[6].strip()
                try:
                    # Parse time HH:MM:SS
                    t = datetime.strptime(time_str, "%H:%M:%S").time()
                    # Use today's date so it matches the video events!
                    today = datetime.now(timezone.utc).date()
                    ts = datetime.combine(today, t).replace(tzinfo=timezone.utc)
                    basket = Decimal(value_str)
                except (ValueError, InvalidOperation):
                    continue
                records.append(
                    PosTransaction(
                        store_id=store_id,
                        transaction_id=transaction_id,
                        timestamp=ts,
                        basket_value_inr=basket,
                    )
                )
        if records:
            db.add_all(records)
            db.commit()
            log.info("pos_seed_done", rows=len(records))
    except Exception as exc:
        db.rollback()
        log.error("pos_seed_error", error=str(exc))
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Lifespan — create tables, seed data
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _STARTUP_TS
    _STARTUP_TS = time.time()

    log.info("startup_begin")
    # Create all tables (idempotent)
    engine, _ = _get_engine()
    Base.metadata.create_all(bind=engine)
    log.info("tables_created")

    # Seed POS data
    _seed_pos_data()

    yield  # application runs here

    log.info("shutdown")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Store Intelligence System",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Structured-logging middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def logging_middleware(request: Request, call_next: Any) -> Response:
    trace_id = str(uuid.uuid4())
    request.state.trace_id = trace_id

    # Try to extract store_id from path (e.g. /stores/STORE_BLR_002/metrics)
    store_id: str | None = None
    parts = request.url.path.strip("/").split("/")
    if len(parts) >= 2 and parts[0] == "stores":
        store_id = parts[1]

    start = time.perf_counter()
    response: Response = await call_next(request)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    log.info(
        "http_request",
        trace_id=trace_id,
        store_id=store_id,
        endpoint=request.url.path,
        method=request.method,
        latency_ms=latency_ms,
        status_code=response.status_code,
    )
    response.headers["X-Trace-Id"] = trace_id
    return response


# ---------------------------------------------------------------------------
# Global exception handler — graceful degradation
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    log.error("unhandled_exception", error=str(exc), path=request.url.path)
    return JSONResponse(
        status_code=503,
        content={"error": "service_unavailable", "detail": str(exc)},
    )


# ---------------------------------------------------------------------------
# Register routers
# ---------------------------------------------------------------------------
from app.ingestion import router as ingestion_router  # noqa: E402
from app.metrics import router as metrics_router  # noqa: E402
from app.funnel import router as funnel_router  # noqa: E402
from app.anomalies import router as anomalies_router  # noqa: E402
from app.health import router as health_router  # noqa: E402

app.include_router(ingestion_router)
app.include_router(metrics_router)
app.include_router(funnel_router)
app.include_router(anomalies_router)
app.include_router(health_router)


def get_startup_ts() -> float:
    """Expose startup timestamp for the health endpoint."""
    return _STARTUP_TS
