"""
pipeline/config.py
==================
Centralised configuration for the Store Intelligence detection pipeline.

All tuneable knobs live here so that operators can adjust behaviour without
touching detection or tracking code.  Values can be overridden via
environment variables where noted.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model & detection
# ---------------------------------------------------------------------------
MODEL_PATH: str = os.getenv("SIS_MODEL_PATH", "yolov8n.pt")
"""Path to the YOLOv8 weights file.  The default ``yolov8n.pt`` is auto-
downloaded by *ultralytics* on first run."""

CONFIDENCE_THRESHOLD: float = float(os.getenv("SIS_CONFIDENCE", "0.45"))
"""Minimum YOLO confidence to keep a detection.  Intentionally low — we
prefer recall over precision and let the tracker sort things out."""

IOU_THRESHOLD: float = float(os.getenv("SIS_IOU", "0.45"))
"""NMS IoU threshold used inside YOLO inference."""

# ---------------------------------------------------------------------------
# Video / timing
# ---------------------------------------------------------------------------
FPS: int = int(os.getenv("SIS_FPS", "15"))
"""Assumed / target frames-per-second for the input clip."""

DWELL_INTERVAL_MS: int = int(os.getenv("SIS_DWELL_INTERVAL_MS", "30000"))
"""Emit a ``ZONE_DWELL`` event every *this many* milliseconds while a
visitor remains inside the same zone."""

# ---------------------------------------------------------------------------
# Staff uniform detection (HSV colour range)
# ---------------------------------------------------------------------------
# Default: full black uniform (black shirt and black pants).
STAFF_UNIFORM_HSV_LOWER: np.ndarray = np.array(
    json.loads(os.getenv("SIS_HSV_LOWER", "[0, 0, 0]")),
    dtype=np.uint8,
)
STAFF_UNIFORM_HSV_UPPER: np.ndarray = np.array(
    json.loads(os.getenv("SIS_HSV_UPPER", "[179, 255, 60]")),
    dtype=np.uint8,
)

STAFF_UNIFORM_RATIO: float = float(os.getenv("SIS_STAFF_RATIO", "0.55"))
"""Fraction of the upper-body crop that must fall inside the HSV range for
the person to be classified as staff."""

# ---------------------------------------------------------------------------
# Re-identification / cross-camera matching
# ---------------------------------------------------------------------------
# Cross-camera Re-ID (Distance threshold. Higher = looser match)
# Using PyTorch Neural Network, deep features are highly correlated for humans, so we need a very tight threshold.
REID_DISTANCE_THRESHOLD: float = float(
    os.getenv("SIS_REID_DISTANCE", "0.45")
)
"""Maximum cosine distance (1 − similarity) to consider two appearance
histograms as the same person across cameras."""

CENTROID_HISTORY_LEN: int = int(os.getenv("SIS_CENTROID_HISTORY", "10"))
"""Number of recent centroids to keep per track for direction estimation."""

# ---------------------------------------------------------------------------
# API / event emission
# ---------------------------------------------------------------------------
API_BASE_URL: str = os.getenv("SIS_API_URL", "http://localhost:8000")
"""Base URL of the Store Intelligence REST API."""

BATCH_SIZE: int = int(os.getenv("SIS_BATCH_SIZE", "5"))
"""Number of events to accumulate before flushing to the API in one POST."""

API_TIMEOUT_S: float = float(os.getenv("SIS_API_TIMEOUT", "10"))
"""HTTP timeout in seconds for the event-ingest endpoint."""

# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------
MAX_AGE: int = int(os.getenv("SIS_MAX_AGE", "900"))
"""Number of frames a track can survive without a matching detection before
it is considered lost (and an EXIT event is emitted)."""

MIN_HITS: int = int(os.getenv("SIS_MIN_HITS", "3"))
"""Minimum number of consecutive hits before a track is considered
confirmed and events start being emitted for it."""

IOU_MATCH_THRESHOLD: float = float(
    os.getenv("SIS_IOU_MATCH_THRESH", "0.3")
)
"""IoU threshold for the Hungarian-algorithm matching step inside the
simple tracker."""

# ---------------------------------------------------------------------------
# Store layout helper
# ---------------------------------------------------------------------------

# Type alias for a zone polygon: list of (x, y) vertices.
ZonePolygon = List[Tuple[float, float]]


def _default_layout() -> Dict[str, Any]:
    """Return a minimal layout structure so the pipeline never crashes when
    ``store_layout.json`` is missing."""
    return {
        "zones": {},          # zone_id -> {"polygon": [...], "label": "..."}
        "cameras": {},        # camera_id -> {"zones": [...]}
        "entry_threshold_y": None,  # y-coordinate used for in/out direction
        "open_hours": {"open": "09:00", "close": "21:00"},
    }


def load_store_layout(path: str) -> Dict[str, Any]:
    """Parse ``store_layout.json`` and return a normalised dict.

    Expected JSON schema (simplified)::

        {
          "zones": {
            "zone_a": {
              "polygon": [[x1,y1], [x2,y2], ...],
              "label": "Skincare Aisle"
            },
            ...
          },
          "cameras": {
            "cam_01": {"zones": ["zone_a", "zone_b"]}
          },
          "entry_threshold_y": 600,
          "billing_zone_id": "billing",
          "open_hours": {"open": "09:00", "close": "21:00"}
        }

    Parameters
    ----------
    path:
        Filesystem path to the JSON file.

    Returns
    -------
    dict
        Normalised layout dictionary.  Falls back to a safe default when the
        file is missing or unparseable.
    """
    layout_path = Path(path)
    if not layout_path.is_file():
        logger.warning(
            "Store layout not found at %s — using empty default layout.",
            path,
        )
        return _default_layout()

    try:
        with open(layout_path, "r", encoding="utf-8") as fh:
            raw: Dict[str, Any] = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to parse store layout: %s", exc)
        return _default_layout()

    # Normalise — fill in any missing top-level keys with safe defaults.
    defaults = _default_layout()
    for key, default_val in defaults.items():
        raw.setdefault(key, default_val)

    # Convert polygon lists to tuples for immutability.
    zones: Dict[str, Any] = raw.get("zones", {})
    for zone_id, zone_data in zones.items():
        polygon_raw = zone_data.get("polygon", [])
        zone_data["polygon"] = [tuple(pt) for pt in polygon_raw]

    logger.info(
        "Loaded store layout with %d zone(s) from %s",
        len(zones),
        path,
    )
    return raw
