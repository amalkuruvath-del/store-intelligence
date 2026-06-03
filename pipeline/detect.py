"""
pipeline/detect.py
==================
Main detection entry-point for the Store Intelligence pipeline.

Usage::

    python -m pipeline.detect \\
        --video  data/clip_01.mp4 \\
        --store-id  STORE_42 \\
        --camera-id  CAM_ENTRANCE \\
        --layout  store_layout.json \\
        --start-time  2026-05-29T10:00:00Z

The script:
1. Loads a YOLOv8 model (auto-downloads ``yolov8n.pt`` if needed).
2. Parses the store layout for zone polygons.
3. Reads frames from the video clip.
4. Detects persons (YOLO class 0), passes them to ``VisitorTracker``.
5. Emits structured events via ``EventEmitter``.
6. Flushes remaining events and emits EXIT for any still-tracked visitors
   when the video ends.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from pipeline import config
from pipeline.emit import EventEmitter
from pipeline.tracker import VisitorTracker, point_in_polygon

logger = logging.getLogger("pipeline.detect")

# YOLO class index for "person"
_PERSON_CLASS = 0

def apply_iom_nms(dets, confs, iom_threshold=0.8):
    if len(dets) == 0:
        return dets, confs
    
    keep = []
    order = np.argsort(confs)[::-1]
    
    for i in range(len(order)):
        idx1 = order[i]
        if idx1 == -1:
            continue
        keep.append(idx1)
        box1 = dets[idx1]
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        
        for j in range(i + 1, len(order)):
            idx2 = order[j]
            if idx2 == -1:
                continue
            box2 = dets[idx2]
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            
            xA = max(box1[0], box2[0])
            yA = max(box1[1], box2[1])
            xB = min(box1[2], box2[2])
            yB = min(box1[3], box2[3])
            
            interArea = max(0, xB - xA) * max(0, yB - yA)
            minArea = min(area1, area2)
            
            if minArea > 0:
                iom = interArea / minArea
                if iom > iom_threshold:
                    order[j] = -1 # Suppress
                    
    return np.array(dets)[keep], np.array(confs)[keep]


# ── CLI ────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run person detection on a CCTV clip and emit events."
    )
    p.add_argument(
        "--video",
        required=True,
        help="Path to the input video clip.",
    )
    p.add_argument(
        "--store-id",
        required=True,
        help="Store identifier string.",
    )
    p.add_argument(
        "--camera-id",
        required=True,
        help="Camera identifier string.",
    )
    p.add_argument(
        "--layout",
        default="store_layout.json",
        help="Path to store_layout.json (default: store_layout.json).",
    )
    p.add_argument(
        "--start-time",
        required=True,
        help=(
            "ISO-8601 timestamp for the first frame of the clip "
            "(e.g. 2026-05-29T10:00:00Z)."
        ),
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return p.parse_args(argv)


def _parse_start_time(raw: str) -> datetime:
    """Parse an ISO-8601 timestamp, defaulting to UTC if no offset."""
    # Python 3.11+ handles trailing Z; older versions do not.
    raw = raw.replace("Z", "+00:00")
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# ── Main pipeline ─────────────────────────────────────────────────────────

def run(
    video_path: str,
    store_id: str,
    camera_id: str,
    layout_path: str,
    start_time: datetime,
) -> None:
    """Execute the full detection → tracking → event-emission pipeline."""

    # ── 1. Load YOLO model ────────────────────────────────────────────
    logger.info("Loading YOLOv8 model: %s", config.MODEL_PATH)
    try:
        from ultralytics import YOLO  # type: ignore[import-untyped]
    except ImportError:
        logger.critical(
            "ultralytics is not installed.  "
            "Run:  pip install ultralytics"
        )
        sys.exit(1)

    model = YOLO(config.MODEL_PATH)

    # ── 2. Load store layout ──────────────────────────────────────────
    layout = config.load_store_layout(layout_path)
    zones = layout.get("zones", {})
    entry_threshold_y = layout.get("entry_threshold_y")
    billing_zone_id = layout.get("billing_zone_id")

    # If the layout specifies which zones this camera sees, restrict.
    cam_cfg = layout.get("cameras", {}).get(camera_id, {})
    visible_zone_ids = cam_cfg.get("zones")
    if visible_zone_ids is not None:
        zones = {
            zid: zdata
            for zid, zdata in zones.items()
            if zid in visible_zone_ids
        }
        logger.info(
            "Camera %s sees zones: %s",
            camera_id,
            list(zones.keys()),
        )

    # ── 3. Tracker ────────────────────────────────────────────────────
    is_billing_camera = False
    billing_zones = [z for z, data in layout.get("zones", {}).items() if "billing" in data.get("label", "").lower()]
    billing_zone_id = billing_zones[0] if billing_zones else None
    
    camera_cfg = layout.get("cameras", {}).get(camera_id, {})
    if camera_cfg:
        cam_zones = camera_cfg.get("zones", [])
        if billing_zone_id in cam_zones:
            is_billing_camera = True

    tracker = VisitorTracker(
        camera_id=camera_id,
        zones=zones,
        entry_threshold_y=layout.get("entry_threshold_y"),
        billing_zone_id=billing_zone_id,
        is_billing_camera=is_billing_camera,
    )
    emitter = EventEmitter(
        store_id=store_id,
        camera_id=camera_id,
    )

    # ── 4. Open video ─────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.critical("Cannot open video: %s", video_path)
        sys.exit(1)

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = config.FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(
        "Opened %s — %.1f fps, ~%d frames",
        video_path,
        video_fps,
        total_frames,
    )

    frame_duration = timedelta(seconds=1.0 / video_fps)
    frame_number = 0
    last_frame_time = start_time

    event_buffer = []
    import time
    last_flush_time = time.time()

    # ── 5. Pass 1: Staff Color Discovery ──────────────────────────────
    max_frames_pass1 = 3 * 60 * video_fps
    logger.info("--- PASS 1: Discovering Staff Color (Max 3 minutes) ---")
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            if frame_number > max_frames_pass1: break
            
            frame_time = start_time + frame_number * frame_duration
            last_frame_time = frame_time

            results = model(frame, conf=0.65, iou=config.IOU_THRESHOLD, classes=[0, 62, 63], verbose=False)
            detections = np.empty((0, 4), dtype=np.float32)
            confidences = np.empty((0,), dtype=np.float32)
            laptop_boxes = np.empty((0, 4), dtype=np.float32)

            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    raw_dets = result.boxes.xyxy.cpu().numpy().astype(np.float32)
                    raw_confs = result.boxes.conf.cpu().numpy().astype(np.float32)
                    raw_cls = result.boxes.cls.cpu().numpy().astype(int)
                    camera_cfg = layout.get("cameras", {}).get(camera_id, {})
                    roi = camera_cfg.get("roi") or layout.get("roi")
                    frame_area = frame.shape[0] * frame.shape[1]
                    
                    filtered_dets, filtered_confs, lap_boxes = [], [], []
                    for d, c, cl in zip(raw_dets, raw_confs, raw_cls):
                        x1, y1, x2, y2 = d
                        if cl == 62 or cl == 63:
                            lap_boxes.append(d)
                            continue
                        width, height = x2 - x1, y2 - y1
                        if height > 0 and (width / height) > 1.2: continue
                        if (width * height) > (0.4 * frame_area): continue
                        foot = ((x1 + x2) / 2.0, y2)
                        if roi and not point_in_polygon(foot, roi): continue
                        filtered_dets.append(d)
                        filtered_confs.append(c)
                        
                    if filtered_dets:
                        detections = np.array(filtered_dets, dtype=np.float32)
                        confidences = np.array(filtered_confs, dtype=np.float32)
                        detections, confidences = apply_iom_nms(detections, confidences, iom_threshold=0.8)
                    if lap_boxes:
                        laptop_boxes = np.array(lap_boxes, dtype=np.float32)

            tracker.update(detections, confidences, frame, frame_time, laptop_boxes=laptop_boxes)
            frame_number += 1
            if frame_number % 500 == 0:
                logger.info("Pass 1: Processed %d frames", frame_number)

    except KeyboardInterrupt:
        logger.warning("Pass 1 interrupted.")

    VisitorTracker.update_staff_colors()

    # ── 6. Pass 2: Hardcoded Event Emission ───────────────────────────
    logger.info("Starting Pass 2 for actual emission...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    VisitorTracker.reset()
    tracker = VisitorTracker(
        camera_id=camera_id,
        zones=zones,
        entry_threshold_y=layout.get("entry_threshold_y"),
        billing_zone_id=billing_zone_id,
        is_billing_camera=is_billing_camera,
    )
    frame_number = 0
    event_buffer = []
    last_flush_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            frame_time = start_time + frame_number * frame_duration
            last_frame_time = frame_time

            results = model(frame, conf=0.65, iou=config.IOU_THRESHOLD, classes=[0, 62, 63], verbose=False)
            detections = np.empty((0, 4), dtype=np.float32)
            confidences = np.empty((0,), dtype=np.float32)
            laptop_boxes = np.empty((0, 4), dtype=np.float32)

            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None and len(result.boxes) > 0:
                    raw_dets = result.boxes.xyxy.cpu().numpy().astype(np.float32)
                    raw_confs = result.boxes.conf.cpu().numpy().astype(np.float32)
                    raw_cls = result.boxes.cls.cpu().numpy().astype(int)
                    camera_cfg = layout.get("cameras", {}).get(camera_id, {})
                    roi = camera_cfg.get("roi") or layout.get("roi")
                    frame_area = frame.shape[0] * frame.shape[1]
                    
                    filtered_dets, filtered_confs, lap_boxes = [], [], []
                    for d, c, cl in zip(raw_dets, raw_confs, raw_cls):
                        x1, y1, x2, y2 = d
                        if cl == 62 or cl == 63:
                            lap_boxes.append(d)
                            continue
                        width, height = x2 - x1, y2 - y1
                        if height > 0 and (width / height) > 1.2: continue
                        if (width * height) > (0.4 * frame_area): continue
                        foot = ((x1 + x2) / 2.0, y2)
                        if roi and not point_in_polygon(foot, roi): continue
                        filtered_dets.append(d)
                        filtered_confs.append(c)
                        
                    if filtered_dets:
                        detections = np.array(filtered_dets, dtype=np.float32)
                        confidences = np.array(filtered_confs, dtype=np.float32)
                        detections, confidences = apply_iom_nms(detections, confidences, iom_threshold=0.8)
                    if lap_boxes:
                        laptop_boxes = np.array(lap_boxes, dtype=np.float32)

            events = tracker.update(detections, confidences, frame, frame_time, laptop_boxes=laptop_boxes)
            if events:
                event_buffer.extend(events)

            # Flush every 2 minutes of real time
            now = time.time()
            if now - last_flush_time >= 120.0:
                for evt in event_buffer:
                    emitter.emit(
                        event_type=evt["event_type"],
                        visitor_id=evt["visitor_id"],
                        timestamp=evt["timestamp"],
                        zone_id=evt.get("zone_id"),
                        dwell_ms=evt.get("dwell_ms", 0),
                        is_staff=evt.get("is_staff", False),
                        confidence=evt.get("confidence", 0.0),
                        metadata=evt.get("metadata", {}),
                    )
                emitter.flush()
                event_buffer.clear()
                last_flush_time = now

            frame_number += 1
            if frame_number % 500 == 0:
                pct = (frame_number / total_frames * 100) if total_frames > 0 else 0
                logger.info("Processed %d / %d frames (%.1f%%)", frame_number, total_frames, pct)

    except KeyboardInterrupt:
        logger.warning("Interrupted by user at frame %d.", frame_number)
    finally:
        cap.release()

    # ── 7. Finalize ───────────────────────────────────────────────────
    final_events = tracker.finalize(last_frame_time)
    event_buffer.extend(final_events)
    
    for evt in event_buffer:
        emitter.emit(
            event_type=evt["event_type"],
            visitor_id=evt["visitor_id"],
            timestamp=evt["timestamp"],
            zone_id=evt.get("zone_id"),
            dwell_ms=evt.get("dwell_ms", 0),
            is_staff=evt.get("is_staff", False),
            confidence=evt.get("confidence", 0.0),
            metadata=evt.get("metadata", {}),
        )
    emitter.flush()

    stats = emitter.stats
    logger.info(
        "Pipeline complete — %d frames processed, %d events emitted, "
        "%d events failed, %d still pending.",
        frame_number,
        stats["emitted"],
        stats["failed"],
        stats["pending"],
    )


# ── Entry-point ───────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(name)-24s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    start_time = _parse_start_time(args.start_time)
    logger.info(
        "Starting pipeline — store=%s  cam=%s  video=%s  t₀=%s",
        args.store_id,
        args.camera_id,
        args.video,
        start_time.isoformat(),
    )

    run(
        video_path=args.video,
        store_id=args.store_id,
        camera_id=args.camera_id,
        layout_path=args.layout,
        start_time=start_time,
    )


if __name__ == "__main__":
    main()
