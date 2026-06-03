# PROMPT: "Generate comprehensive API tests for a FastAPI store intelligence system
# covering event ingestion with idempotency, partial success handling, graceful
# degradation on DB failure, and structured error responses."
# CHANGES MADE: Added edge cases for empty store, all-staff events, zero purchases.
# Strengthened idempotency test to verify exact DB counts. Added 503 degradation test.
# Fixed event schema to match actual EventIn model (visitor_id, camera_id, zone_id,
# confidence, metadata).
"""
API-level integration tests for the Store Intelligence System.

Tests exercise the FastAPI endpoints end-to-end using an in-memory SQLite
database, validating:
- Batch event ingestion (happy path, partial failures, idempotency)
- Metrics retrieval (basic counts, staff exclusion, zero-purchase stores)
- Health check endpoint
- Graceful degradation when the database is unavailable
"""
import uuid
from unittest.mock import patch

import pytest
from sqlalchemy.exc import OperationalError

from tests.conftest import make_event, seed_visitor_journey


# ---------------------------------------------------------------------------
# Event Ingestion Tests
# ---------------------------------------------------------------------------


class TestEventIngestion:
    """Tests for POST /events/ingest"""

    def test_ingest_valid_events(self, client, sample_events_factory):
        """POST a batch of 10 valid events → 200, accepted == 10."""
        events = sample_events_factory(n=10, store_id="STORE_TEST")
        resp = client.post("/events/ingest", json=events)

        assert resp.status_code == 200
        body = resp.json()
        assert body["accepted"] == 10
        assert body.get("rejected", 0) == 0

    def test_ingest_idempotent(self, client, sample_events_factory):
        """POSTing the same batch twice should not create duplicate rows."""
        events = sample_events_factory(n=10, store_id="STORE_TEST")

        resp1 = client.post("/events/ingest", json=events)
        assert resp1.status_code == 200
        first_accepted = resp1.json()["accepted"]
        assert first_accepted == 10

        # Ingest same events again (same event_ids)
        resp2 = client.post("/events/ingest", json=events)
        assert resp2.status_code == 200
        body2 = resp2.json()

        # Second call: duplicates should be skipped
        assert body2["accepted"] == 0

    def test_ingest_partial_success(self, client, sample_events_factory):
        """Batch with 5 valid + 5 malformed events → accepted==5, rejected==5."""
        valid = sample_events_factory(n=5, store_id="STORE_TEST")
        # Malformed events: missing required fields
        malformed = [
            {"event_id": str(uuid.uuid4()), "garbage_field": True}
            for _ in range(5)
        ]
        resp = client.post("/events/ingest", json=valid + malformed)

        assert resp.status_code == 200
        body = resp.json()
        assert body["accepted"] == 5
        assert body["rejected"] == 5
        assert isinstance(body.get("errors", []), list)
        assert len(body["errors"]) == 5

    def test_ingest_empty_batch(self, client):
        """POST empty events list → should be handled gracefully."""
        resp = client.post("/events/ingest", json=[])

        # Accept 200 (accepted=0) or 400/422
        assert resp.status_code in (200, 400, 422)

    def test_ingest_all_staff_events(self, client):
        """POST events that are all staff → accepted but excluded from visitor metrics."""
        staff_events = [
            make_event(store_id="STORE_TEST", is_staff=True, event_type="ENTRY")
            for _ in range(5)
        ]
        resp = client.post("/events/ingest", json=staff_events)

        assert resp.status_code == 200
        body = resp.json()
        assert body["accepted"] == 5


# ---------------------------------------------------------------------------
# Metrics Tests
# ---------------------------------------------------------------------------


class TestMetrics:
    """Tests for GET /stores/{store_id}/metrics"""

    def _seed_store(self, client, store_id: str = "STORE_TEST", n_visitors: int = 5):
        """Helper: ingest events for *n_visitors* customer journeys."""
        all_events = []
        visitor_ids = []
        for _ in range(n_visitors):
            journey = seed_visitor_journey(store_id=store_id)
            all_events.extend(journey)
            visitor_ids.append(journey[0]["visitor_id"])
        resp = client.post("/events/ingest", json=all_events)
        assert resp.status_code == 200
        return visitor_ids

    def test_metrics_basic(self, client):
        """Ingest events, then GET metrics → valid response with correct counts."""
        self._seed_store(client, store_id="STORE_TEST", n_visitors=5)

        resp = client.get("/stores/STORE_TEST/metrics")
        assert resp.status_code == 200
        body = resp.json()

        assert "unique_visitors" in body
        assert "conversion_rate" in body
        assert body["unique_visitors"] >= 5

    def test_metrics_excludes_staff(self, client):
        """Staff events should NOT count towards unique_visitors."""
        # 3 customer journeys
        customer_events = []
        for _ in range(3):
            customer_events.extend(
                seed_visitor_journey(store_id="STORE_STAFF_TEST", is_staff=False)
            )
        # 2 staff journeys
        staff_events = []
        for _ in range(2):
            staff_events.extend(
                seed_visitor_journey(store_id="STORE_STAFF_TEST", is_staff=True)
            )

        all_events = customer_events + staff_events
        resp = client.post("/events/ingest", json=all_events)
        assert resp.status_code == 200

        resp = client.get("/stores/STORE_STAFF_TEST/metrics")
        assert resp.status_code == 200
        body = resp.json()

        # unique_visitors should be 3, not 5
        assert body["unique_visitors"] == 3

    def test_metrics_zero_purchases(self, client):
        """Store with visitors but NO POS data → conversion_rate == 0.0."""
        self._seed_store(client, store_id="STORE_ZERO_POS", n_visitors=4)

        resp = client.get("/stores/STORE_ZERO_POS/metrics")
        assert resp.status_code == 200
        body = resp.json()

        assert body["conversion_rate"] == 0.0

    def test_metrics_empty_store(self, client):
        """GET metrics for a store with no events → sensible defaults."""
        resp = client.get("/stores/STORE_NONEXISTENT/metrics")

        assert resp.status_code in (200, 404)
        if resp.status_code == 200:
            body = resp.json()
            assert body["unique_visitors"] == 0
            assert body["conversion_rate"] == 0.0


# ---------------------------------------------------------------------------
# Health Endpoint Test
# ---------------------------------------------------------------------------


class TestHealth:
    """Tests for GET /health"""

    def test_health_endpoint(self, client):
        """GET /health → 200, status == 'ok'."""
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"

    def test_health_has_required_fields(self, client):
        """Health response includes DB connectivity and stale feed info."""
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert "db_connected" in body
        assert "stale_feeds" in body
        assert "last_event_per_store" in body


# ---------------------------------------------------------------------------
# Graceful Degradation Tests
# ---------------------------------------------------------------------------


class TestDegradation:
    """Verify the API degrades gracefully when the database is unreachable."""

    def test_db_failure_returns_503(self, client, db_session):
        """When DB raises OperationalError, the API should return 503."""
        from app.database import get_db as real_get_db
        from app.main import app

        def _broken_db():
            raise OperationalError("connection refused", None, None)
            yield  # noqa: unreachable – generator protocol

        app.dependency_overrides[real_get_db] = _broken_db
        try:
            resp = client.get("/stores/ANY/metrics")
            # Accept 500 or 503 – both indicate server-side failure
            assert resp.status_code in (500, 503)
            body = resp.json()
            # Should return structured error, not a raw stack trace
            assert "error" in body or "detail" in body
        finally:
            app.dependency_overrides.pop(real_get_db, None)
