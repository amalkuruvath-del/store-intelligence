"""
pipeline/emit.py
================
Event construction and batched API communication for the Store Intelligence
detection pipeline.

``EventEmitter`` accumulates structured event dicts in an internal buffer
and flushes them to the REST API when the buffer reaches ``batch_size`` or
when explicitly asked.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

from pipeline import config

logger = logging.getLogger(__name__)

# ── Canonical event-type constants ─────────────────────────────────────────
ENTRY = "ENTRY"
EXIT = "EXIT"
REENTRY = "REENTRY"
ZONE_ENTER = "ZONE_ENTER"
ZONE_EXIT = "ZONE_EXIT"
ZONE_DWELL = "ZONE_DWELL"
BILLING_QUEUE_JOIN = "BILLING_QUEUE_JOIN"
BILLING_QUEUE_LEAVE = "BILLING_QUEUE_LEAVE"


class EventEmitter:
    """Batched event emitter that POSTs to the ingest endpoint.

    Parameters
    ----------
    store_id:
        Identifier of the retail store being monitored.
    camera_id:
        Identifier of the specific camera / clip source.
    api_url:
        Base URL of the REST API (e.g. ``http://localhost:8000``).
    batch_size:
        Number of events to accumulate before auto-flushing.
    """

    def __init__(
        self,
        store_id: str,
        camera_id: str,
        api_url: str = config.API_BASE_URL,
        batch_size: int = config.BATCH_SIZE,
    ) -> None:
        self.store_id = store_id
        self.camera_id = camera_id
        self.api_url = api_url.rstrip("/")
        self.batch_size = batch_size

        self._buffer: List[Dict[str, Any]] = []
        self._total_emitted: int = 0
        self._total_failed: int = 0

    # ── public API ─────────────────────────────────────────────────────────

    def emit(
        self,
        event_type: str,
        visitor_id: str,
        timestamp: datetime,
        zone_id: Optional[str] = None,
        dwell_ms: int = 0,
        is_staff: bool = False,
        confidence: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create an event dict and add it to the buffer.

        Auto-flushes when the buffer reaches *batch_size*.
        """
        event = self._make_event(
            event_type=event_type,
            visitor_id=visitor_id,
            timestamp=timestamp,
            zone_id=zone_id,
            dwell_ms=dwell_ms,
            is_staff=is_staff,
            confidence=confidence,
            metadata=metadata or {},
        )
        self._buffer.append(event)
        logger.debug(
            "Buffered %s for %s (buffer=%d)",
            event_type,
            visitor_id,
            len(self._buffer),
        )

        if len(self._buffer) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        """POST all buffered events to the ingest endpoint and clear the
        buffer.  Failures are logged but never crash the pipeline."""
        if not self._buffer:
            return

        payload = list(self._buffer)
        count = len(payload)
        self._buffer.clear()

        url = f"{self.api_url}/events/ingest"
        try:
            resp = requests.post(
                url,
                json=payload,
                timeout=config.API_TIMEOUT_S,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            self._total_emitted += count
            logger.info(
                "Flushed %d event(s) → %s [%s]",
                count,
                url,
                resp.status_code,
            )
        except requests.ConnectionError:
            self._total_failed += count
            logger.error(
                "API unreachable at %s — %d event(s) dropped.", url, count
            )
        except requests.HTTPError as exc:
            self._total_failed += count
            logger.error(
                "API returned %s — %d event(s) dropped: %s",
                exc.response.status_code if exc.response is not None else "?",
                count,
                exc,
            )
        except requests.Timeout:
            self._total_failed += count
            logger.error(
                "API timed out after %.1fs — %d event(s) dropped.",
                config.API_TIMEOUT_S,
                count,
            )
        except Exception as exc:  # noqa: BLE001
            self._total_failed += count
            logger.exception("Unexpected error flushing events: %s", exc)

    @property
    def stats(self) -> Dict[str, int]:
        """Return a summary of emitted / failed counts."""
        return {
            "emitted": self._total_emitted,
            "failed": self._total_failed,
            "pending": len(self._buffer),
        }

    # ── internals ──────────────────────────────────────────────────────────

    def _make_event(
        self,
        event_type: str,
        visitor_id: str,
        timestamp: datetime,
        zone_id: Optional[str],
        dwell_ms: int,
        is_staff: bool,
        confidence: float,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build a single event dict conforming to the ingest schema."""
        ts_str = timestamp.isoformat(timespec="milliseconds")
        if not ts_str.endswith("Z"):
            # Ensure UTC suffix; strip any offset first for safety.
            ts_str = ts_str.replace("+00:00", "") + "Z"

        return {
            "event_id": str(uuid.uuid4()),
            "store_id": self.store_id,
            "camera_id": self.camera_id,
            "visitor_id": visitor_id,
            "event_type": event_type,
            "timestamp": ts_str,
            "zone_id": zone_id,
            "dwell_ms": dwell_ms,
            "is_staff": bool(is_staff),
            "confidence": round(float(confidence), 4),
            "metadata": metadata,
        }
