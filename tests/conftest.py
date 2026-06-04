"""
Shared test fixtures for the Store Intelligence System test suite.

Provides:
- In-memory SQLite database engine and session factory
- FastAPI TestClient with overridden get_db dependency
- Factory functions to generate valid EventIn dicts and POS transactions
- Utility helpers for seeding test data

NOTE: Uses SQLite in-memory for fast, isolated tests.  The production API
uses PostgreSQL, but SQLAlchemy ORM compatibility keeps them interchangeable.
"""
import os
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
import sys
import uuid
import random
from datetime import datetime, timedelta, timezone
from typing import Generator, List, Dict, Any, Optional

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event as sa_event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

# ---------------------------------------------------------------------------
# Adjust import paths – the app package lives one level up from tests/
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.main import app  # noqa: E402  FastAPI application instance
from app.database import Base, get_db  # noqa: E402  ORM base & dependency


# ============================= DATABASE FIXTURES =============================

SQLALCHEMY_TEST_DATABASE_URL = "sqlite:///:memory:"


@pytest.fixture(scope="session")
def db_engine():
    """Create a single in-memory SQLite engine shared across the test session."""
    engine = create_engine(
        SQLALCHEMY_TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @sa_event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture(scope="function")
def db_session(db_engine) -> Generator[Session, None, None]:
    """Provide a transactional database session that rolls back after each test."""
    TestingSessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=db_engine
    )
    connection = db_engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture(scope="function")
def client(db_session) -> Generator[TestClient, None, None]:
    """FastAPI TestClient with the database dependency overridden."""

    def _override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    app.dependency_overrides.clear()


# ========================== EVENT FACTORY HELPERS ============================

# These match the exact Pydantic EventIn schema defined in app/models.py
EVENT_TYPE_CHOICES = [
    "ENTRY", "EXIT", "ZONE_ENTER", "ZONE_EXIT", "ZONE_DWELL",
    "BILLING_QUEUE_JOIN", "BILLING_QUEUE_ABANDON", "REENTRY",
]
ZONE_NAMES = ["SKINCARE", "COSMETICS", "HAIRCARE", "BILLING", "ENTRANCE"]
CAMERA_IDS = ["CAM_ENTRY_01", "CAM_FLOOR_01", "CAM_BILLING_01"]


def _base_timestamp(offset_minutes: int = 0) -> str:
    """Return an ISO-8601 UTC timestamp offset from 'now' by *offset_minutes*."""
    dt = datetime.now(timezone.utc) - timedelta(minutes=offset_minutes)
    return dt.isoformat()


def make_event(
    *,
    store_id: str = "STORE_TEST",
    event_type: str = "ENTRY",
    visitor_id: Optional[str] = None,
    camera_id: str = "CAM_ENTRY_01",
    zone_id: Optional[str] = None,
    timestamp: Optional[str] = None,
    event_id: Optional[str] = None,
    is_staff: bool = False,
    confidence: float = 0.85,
    dwell_ms: int = 0,
    queue_depth: Optional[int] = None,
    sku_zone: Optional[str] = None,
    session_seq: int = 1,
) -> Dict[str, Any]:
    """Build a single valid event dict matching the EventIn Pydantic schema.

    All fields align with app/models.py::EventIn.
    """
    if visitor_id is None:
        visitor_id = f"VIS_{uuid.uuid4().hex[:8]}"
    if timestamp is None:
        timestamp = _base_timestamp(offset_minutes=random.randint(0, 120))
    if event_id is None:
        event_id = str(uuid.uuid4())

    # Set sensible zone_id defaults based on event type
    if zone_id is None and event_type not in ("ENTRY", "EXIT", "REENTRY"):
        zone_id = random.choice(ZONE_NAMES[:3])  # non-billing zones
    if event_type in ("BILLING_QUEUE_JOIN", "BILLING_QUEUE_ABANDON"):
        zone_id = "BILLING"

    return {
        "event_id": event_id,
        "store_id": store_id,
        "camera_id": camera_id,
        "visitor_id": visitor_id,
        "event_type": event_type,
        "timestamp": timestamp,
        "zone_id": zone_id,
        "dwell_ms": dwell_ms,
        "is_staff": is_staff,
        "confidence": confidence,
        "metadata": {
            "queue_depth": queue_depth,
            "sku_zone": sku_zone or zone_id,
            "session_seq": session_seq,
        },
    }


@pytest.fixture()
def sample_events_factory():
    """Return a callable that produces a list of *n* valid event dicts."""

    def _factory(
        n: int = 10,
        store_id: str = "STORE_TEST",
        event_type: str = "ENTRY",
        is_staff: bool = False,
    ) -> List[Dict[str, Any]]:
        return [
            make_event(
                store_id=store_id,
                event_type=event_type,
                is_staff=is_staff,
                session_seq=i + 1,
            )
            for i in range(n)
        ]

    return _factory


# ======================== VISITOR JOURNEY FACTORY ============================


def seed_visitor_journey(
    store_id: str = "STORE_TEST",
    visitor_id: Optional[str] = None,
    zones: Optional[List[str]] = None,
    include_billing: bool = True,
    is_staff: bool = False,
) -> List[Dict[str, Any]]:
    """Generate a full visitor journey as a list of event dicts.

    Journey: ENTRY → ZONE_ENTER(zones) → ZONE_DWELL(zones) →
             [BILLING_QUEUE_JOIN] → EXIT
    """
    if visitor_id is None:
        visitor_id = f"VIS_{uuid.uuid4().hex[:8]}"
    if zones is None:
        zones = ["SKINCARE", "COSMETICS"]

    events: List[Dict[str, Any]] = []
    base_time = datetime.now(timezone.utc) - timedelta(hours=1)
    seq = 1

    # ENTRY
    events.append(make_event(
        store_id=store_id,
        event_type="ENTRY",
        visitor_id=visitor_id,
        camera_id="CAM_ENTRY_01",
        zone_id=None,
        timestamp=base_time.isoformat(),
        is_staff=is_staff,
        session_seq=seq,
    ))
    seq += 1

    # Zone visits
    for idx, z in enumerate(zones):
        enter_time = base_time + timedelta(minutes=3 * (idx + 1))
        events.append(make_event(
            store_id=store_id,
            event_type="ZONE_ENTER",
            visitor_id=visitor_id,
            camera_id="CAM_FLOOR_01",
            zone_id=z,
            timestamp=enter_time.isoformat(),
            is_staff=is_staff,
            session_seq=seq,
        ))
        seq += 1

        # Dwell event (30s+ in zone)
        dwell_time = enter_time + timedelta(seconds=35)
        events.append(make_event(
            store_id=store_id,
            event_type="ZONE_DWELL",
            visitor_id=visitor_id,
            camera_id="CAM_FLOOR_01",
            zone_id=z,
            dwell_ms=35000,
            timestamp=dwell_time.isoformat(),
            is_staff=is_staff,
            session_seq=seq,
        ))
        seq += 1

        # Zone exit
        exit_zone_time = enter_time + timedelta(minutes=2)
        events.append(make_event(
            store_id=store_id,
            event_type="ZONE_EXIT",
            visitor_id=visitor_id,
            camera_id="CAM_FLOOR_01",
            zone_id=z,
            timestamp=exit_zone_time.isoformat(),
            is_staff=is_staff,
            session_seq=seq,
        ))
        seq += 1

    # Billing queue
    if include_billing:
        billing_time = base_time + timedelta(minutes=3 * (len(zones) + 1))
        events.append(make_event(
            store_id=store_id,
            event_type="BILLING_QUEUE_JOIN",
            visitor_id=visitor_id,
            camera_id="CAM_BILLING_01",
            zone_id="BILLING",
            timestamp=billing_time.isoformat(),
            is_staff=is_staff,
            queue_depth=random.randint(1, 5),
            session_seq=seq,
        ))
        seq += 1

    # EXIT
    exit_time = base_time + timedelta(minutes=3 * (len(zones) + 2) + 5)
    events.append(make_event(
        store_id=store_id,
        event_type="EXIT",
        visitor_id=visitor_id,
        camera_id="CAM_ENTRY_01",
        zone_id=None,
        timestamp=exit_time.isoformat(),
        is_staff=is_staff,
        session_seq=seq,
    ))

    return events


@pytest.fixture()
def seed_journey():
    """Expose seed_visitor_journey as a fixture callable."""
    return seed_visitor_journey
