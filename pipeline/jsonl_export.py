"""
pipeline/jsonl_export.py
========================
Exports filtered events from the PostgreSQL database into the challenge's
required ``event_log.jsonl`` schema.

Called automatically after every clip finishes processing, so the JSONL
file is **always** in a clean, filtered state — even if the pipeline is
interrupted mid-run.

Filtering logic:
    - Removes false-positive "footpath walkers" — visitors who were only
      ever captured on entry cameras and never appeared deeper in the store.
    - Uses a generic heuristic: any camera whose name contains "entry" is
      treated as an entry-only camera.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import psycopg2

logger = logging.getLogger(__name__)

# Path to the output JSONL file (always in the project root)
JSONL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "event_log.jsonl")

DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/store_intelligence",
)


def _get_connection():
    """Open a fresh psycopg2 connection."""
    return psycopg2.connect(DB_URL)


def _fetch_filtered_events(store_id: str) -> List[Dict[str, Any]]:
    """Query the database for events belonging to *store_id*, filtering
    out visitors who only ever appeared on entry cameras (likely footpath
    walkers caught through the store window).
    """
    conn = _get_connection()
    cur = conn.cursor()

    # Step 1: Find all entry-only cameras for this store
    # (cameras whose name contains 'entry', case-insensitive)
    cur.execute(
        """
        SELECT DISTINCT camera_id FROM events
        WHERE store_id = %s AND LOWER(camera_id) LIKE '%%entry%%'
        """,
        (store_id,),
    )
    entry_cameras = {row[0] for row in cur.fetchall()}

    if entry_cameras:
        # Step 2: Fetch events only for visitors who appeared on at least
        # one non-entry camera (i.e., they actually entered the store).
        placeholders = ", ".join(["%s"] * len(entry_cameras))
        query = f"""
            SELECT * FROM events
            WHERE store_id = %s
              AND visitor_id IN (
                  SELECT DISTINCT visitor_id FROM events
                  WHERE store_id = %s
                    AND camera_id NOT IN ({placeholders})
              )
            ORDER BY timestamp
        """
        params = [store_id, store_id] + list(entry_cameras)
        cur.execute(query, params)
    else:
        # No entry cameras found — return all events for this store
        cur.execute(
            "SELECT * FROM events WHERE store_id = %s ORDER BY timestamp",
            (store_id,),
        )

    cols = [desc[0] for desc in cur.description]
    events = [dict(zip(cols, row)) for row in cur.fetchall()]
    conn.close()
    return events


def _map_event(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Map a database row to the challenge's JSONL schema."""
    e_type = row["event_type"].upper()
    ts = row["timestamp"].isoformat()

    if e_type in ("ENTRY", "EXIT"):
        return {
            "event_type": e_type.lower(),
            "id_token": f"ID_{row['visitor_id']}",
            "store_code": row["store_id"],
            "camera_id": row["camera_id"],
            "event_timestamp": ts,
            "is_staff": row["is_staff"],
            "gender_pred": None,
            "age_pred": None,
            "age_bucket": None,
            "is_face_hidden": False,
            "group_id": None,
            "group_size": None,
        }
    elif e_type in ("ZONE_ENTER", "ZONE_EXIT"):
        return {
            "event_type": "zone_entered" if e_type == "ZONE_ENTER" else "zone_exited",
            "track_id": row["visitor_id"],
            "store_id": row["store_id"],
            "camera_id": row["camera_id"],
            "zone_id": row["zone_id"],
            "zone_name": row["zone_id"],
            "zone_type": "BILLING" if "BILLING" in str(row["zone_id"]).upper() else "SHELF",
            "is_revenue_zone": "Yes",
            "event_time": ts,
            "zone_hotspot_x": 0.0,
            "zone_hotspot_y": 0.0,
            "gender": None,
            "age": None,
            "age_bucket": None,
        }
    return None


def regenerate_jsonl(store_id: str) -> int:
    """Regenerate ``event_log.jsonl`` for the given *store_id*.

    - Reads the existing file and keeps all lines from OTHER stores intact.
    - Queries the database for filtered events for *store_id*.
    - Appends the clean mapped events for *store_id*.

    Returns the number of events written for this store.
    """
    jsonl_path = JSONL_PATH

    # ── Step 1: Read existing lines and keep OTHER stores ──────────────
    other_store_lines: List[str] = []
    if os.path.isfile(jsonl_path):
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    # Events use either "store_code" (entry/exit) or
                    # "store_id" (zone events) depending on the schema.
                    line_store = obj.get("store_code") or obj.get("store_id", "")
                    if line_store != store_id:
                        other_store_lines.append(line)
                except json.JSONDecodeError:
                    # Keep malformed lines to avoid data loss
                    other_store_lines.append(line)

    # ── Step 2: Fetch clean, filtered events from DB ───────────────────
    try:
        db_events = _fetch_filtered_events(store_id)
    except Exception as exc:
        logger.error("Failed to fetch events from DB for %s: %s", store_id, exc)
        print(f"[WARN] Could not regenerate JSONL for {store_id}: {exc}")
        return 0

    # ── Step 3: Map to challenge schema ────────────────────────────────
    mapped_lines: List[str] = []
    for ev in db_events:
        mapped = _map_event(ev)
        if mapped:
            mapped_lines.append(json.dumps(mapped))

    # ── Step 4: Write back (other stores + this store's clean data) ────
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for line in other_store_lines:
            f.write(line + "\n")
        for line in mapped_lines:
            f.write(line + "\n")

    count = len(mapped_lines)
    print(f"[INFO] event_log.jsonl updated: {count} filtered events for {store_id} "
          f"({len(other_store_lines)} events from other stores preserved)")
    return count
