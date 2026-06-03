"""
Pydantic request/response schemas and SQLAlchemy ORM models.

All event types, metadata, and the POS transaction table are defined here.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


# ---------------------------------------------------------------------------
# Pydantic schemas — request / response
# ---------------------------------------------------------------------------


class EventMetadata(BaseModel):
    """Nested metadata embedded in every event."""
    queue_depth: Optional[int] = None
    sku_zone: Optional[str] = None
    session_seq: int


class EventIn(BaseModel):
    """Inbound event payload — one element of the batch array."""
    event_id: str
    store_id: str
    camera_id: str
    visitor_id: str
    event_type: Literal[
        "ENTRY",
        "EXIT",
        "ZONE_ENTER",
        "ZONE_EXIT",
        "ZONE_DWELL",
        "BILLING_QUEUE_JOIN",
        "BILLING_QUEUE_ABANDON",
        "REENTRY",
    ]
    timestamp: datetime
    zone_id: Optional[str] = None
    dwell_ms: int = 0
    is_staff: bool = False
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: EventMetadata


class IngestError(BaseModel):
    """Describes a single rejected event in an ingest batch."""
    index: int
    reason: str


class IngestResponse(BaseModel):
    """Response body for POST /events/ingest."""
    accepted: int
    rejected: int
    errors: List[IngestError] = []


class ZoneDwell(BaseModel):
    """Per-zone average dwell for the metrics endpoint."""
    zone_id: str
    avg_dwell_ms: float


class MetricsResponse(BaseModel):
    """Response body for GET /stores/{store_id}/metrics."""
    store_id: str
    date: str
    unique_visitors: int
    conversion_rate: float
    avg_dwell_per_zone: Dict[str, float]
    current_queue_depth: int
    abandonment_rate: float


class HeatmapCell(BaseModel):
    """One zone in the heatmap response."""
    zone_id: str
    visit_count: int
    avg_dwell_ms: float
    normalized_score: float
    data_confidence: Literal["low", "high"]


class FunnelStage(BaseModel):
    """A single stage of the conversion funnel."""
    stage: str
    count: int
    drop_off_pct: float


class FunnelResponse(BaseModel):
    """Response body for GET /stores/{store_id}/funnel."""
    store_id: str
    stages: List[FunnelStage]


class AnomalyItem(BaseModel):
    """One detected anomaly."""
    type: str
    severity: Literal["INFO", "WARN", "CRITICAL"]
    zone_id: Optional[str] = None
    detail: str
    suggested_action: str
    detected_at: datetime


class AnomalyResponse(BaseModel):
    """Response body for GET /stores/{store_id}/anomalies."""
    store_id: str
    anomalies: List[AnomalyItem]


class HealthResponse(BaseModel):
    """Response body for GET /health."""
    status: Literal["ok", "degraded"]
    db_connected: bool
    last_event_per_store: Dict[str, Optional[str]]
    stale_feeds: List[str]
    uptime_seconds: float


class ErrorResponse(BaseModel):
    """Standard error envelope for 503 / other failures."""
    error: str
    detail: str


# ---------------------------------------------------------------------------
# SQLAlchemy ORM models
# ---------------------------------------------------------------------------


class EventRecord(Base):
    """Persisted camera / sensor event."""
    __tablename__ = "events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    store_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    camera_id: Mapped[str] = mapped_column(String(64), nullable=False)
    visitor_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    zone_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    dwell_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    is_staff: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    meta_queue_depth: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    meta_sku_zone: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    meta_session_seq: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    __table_args__ = (
        Index("ix_events_store_ts", "store_id", "timestamp"),
        Index("ix_events_store_visitor", "store_id", "visitor_id"),
    )


class PosTransaction(Base):
    """Point-of-sale transaction record."""
    __tablename__ = "pos_transactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    store_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    transaction_id: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    basket_value_inr: Mapped[Decimal] = mapped_column(Numeric(12, 2), nullable=False)
