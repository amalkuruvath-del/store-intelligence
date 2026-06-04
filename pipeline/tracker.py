"""
pipeline/tracker.py
===================
Visitor tracking and identity management for the Store Intelligence
detection pipeline.

Provides:
* **SimpleTracker** — an IoU-based multi-object tracker that uses the
  Hungarian algorithm (``scipy.optimize.linear_sum_assignment``) for
  frame-to-frame matching.
* **VisitorTracker** — higher-level wrapper that maintains per-visitor
  state (zone occupancy, dwell timers, staff classification, direction,
  cross-camera re-id) and returns structured events on every ``update()``.
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple
from datetime import datetime
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from pipeline import config
from pipeline.emit import (
    BILLING_QUEUE_JOIN,
    BILLING_QUEUE_ABANDON,
    ENTRY,
    EXIT,
    REENTRY,
    ZONE_DWELL,
    ZONE_ENTER,
    ZONE_EXIT,
)

logger = logging.getLogger(__name__)

# Type aliases
BBox = Tuple[float, float, float, float]  # (x1, y1, x2, y2)
Point = Tuple[float, float]               # (x, y)


# ═══════════════════════════════════════════════════════════════════════════
# Low-level IoU tracker (replaces ByteTrack when it's not available)
# ═══════════════════════════════════════════════════════════════════════════

def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """Compute IoU between two boxes ``[x1, y1, x2, y2]``."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _iou_cost_matrix(
    tracks: List[np.ndarray],
    detections: List[np.ndarray],
) -> np.ndarray:
    """Build an IoU *cost* matrix (1 − IoU) for Hungarian matching."""
    n_tracks = len(tracks)
    n_dets = len(detections)
    cost = np.ones((n_tracks, n_dets), dtype=np.float64)
    for t_idx, t_box in enumerate(tracks):
        for d_idx, d_box in enumerate(detections):
            cost[t_idx, d_idx] = 1.0 - _iou(t_box, d_box)
    return cost

def _init_kcf(frame: np.ndarray, bbox: np.ndarray) -> Any:
    try:
        tracker = cv2.TrackerKCF_create()
    except Exception:
        tracker = None
        
    if tracker is None: return None
    
    x1, y1, x2, y2 = bbox
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    h_f, w_f = frame.shape[:2]
    
    x = max(0, min(int(x1), w_f - int(w)))
    y = max(0, min(int(y1), h_f - int(h)))
    
    try:
        tracker.init(frame, (x, y, int(w), int(h)))
        return tracker
    except Exception:
        return None



@dataclass
class _Track:
    """Internal bookkeeping for a single tracked object."""

    track_id: int
    bbox: np.ndarray            # last matched [x1, y1, x2, y2]
    hits: int = 1               # consecutive frames with a match
    age: int = 0                # frames since last match
    confidence: float = 0.0     # latest detection confidence
    cv2_tracker: Any = None
    tracker_initialized: bool = False
    history: List[Tuple[float, float]] = field(default_factory=list)


class SimpleTracker:
    """Minimalist IoU tracker using the Hungarian algorithm.

    Works well for moderate-density scenes typical of retail stores.
    """

    _next_id: int = 0

    def __init__(
        self,
        iou_threshold: float = config.IOU_MATCH_THRESHOLD,
        max_age: int = config.MAX_AGE,
        min_hits: int = config.MIN_HITS,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self._tracks: List[_Track] = []
        self._last_lost_ids: List[int] = []

    # ------------------------------------------------------------------
    def update(
        self,
        detections: np.ndarray,
        confidences: np.ndarray,
        frame: Optional[np.ndarray] = None,
    ) -> List[Tuple[int, np.ndarray, float]]:
        """Match *detections* to existing tracks and return confirmed tracks.

        Parameters
        ----------
        detections:
            ``(N, 4)`` array of ``[x1, y1, x2, y2]``.
        confidences:
            ``(N,)`` array of detection confidences.

        Returns
        -------
        list of (track_id, bbox, confidence)
            Only tracks with ``hits >= min_hits`` are returned.
        """
        if len(detections) == 0:
            # Age-out existing tracks.
            surviving: List[_Track] = []
            for trk in self._tracks:
                trk.age += 1
                if trk.age <= self.max_age:
                    surviving.append(trk)
            self._tracks = surviving
            return [
                (t.track_id, t.bbox, t.confidence)
                for t in self._tracks
                if t.hits >= self.min_hits
            ]

        det_boxes = [d for d in detections]

        if len(self._tracks) == 0:
            # No existing tracks — initialise from detections.
            for d, c in zip(det_boxes, confidences):
                self._tracks.append(
                    _Track(
                        track_id=self._alloc_id(),
                        bbox=np.array(d, dtype=np.float64),
                        confidence=float(c),
                    )
                )
        else:
            trk_boxes = [t.bbox for t in self._tracks]
            cost = _iou_cost_matrix(trk_boxes, det_boxes)
            row_idx, col_idx = linear_sum_assignment(cost)

            matched_trk: set[int] = set()
            matched_det: set[int] = set()

            for r, c_idx in zip(row_idx, col_idx):
                if cost[r, c_idx] <= (1.0 - self.iou_threshold):
                    # Good match
                    self._tracks[r].bbox = np.array(
                        det_boxes[c_idx], dtype=np.float64
                    )
                    
                    c_x = (det_boxes[c_idx][0] + det_boxes[c_idx][2]) / 2.0
                    c_y = (det_boxes[c_idx][1] + det_boxes[c_idx][3]) / 2.0
                    self._tracks[r].history.append((c_x, c_y))
                    
                    self._tracks[r].hits += 1
                    self._tracks[r].age = 0
                    self._tracks[r].confidence = float(confidences[c_idx])
                    matched_trk.add(r)
                    matched_det.add(c_idx)

            unmatched_t_idx = [i for i in range(len(self._tracks)) if i not in matched_trk]
            unmatched_d_idx = [i for i in range(len(det_boxes)) if i not in matched_det]

            # --- PASS 2: Centroid Distance Fallback ---
            if unmatched_t_idx and unmatched_d_idx:
                def centroid(box):
                    return ((box[0]+box[2])/2.0, (box[1]+box[3])/2.0)
                
                dist_cost = np.ones((len(unmatched_t_idx), len(unmatched_d_idx)), dtype=np.float64) * 1000.0
                for r_i, t_i in enumerate(unmatched_t_idx):
                    for c_i, d_i in enumerate(unmatched_d_idx):
                        c_t = centroid(self._tracks[t_i].bbox)
                        c_d = centroid(det_boxes[d_i])
                        dist = ((c_t[0] - c_d[0])**2 + (c_t[1] - c_d[1])**2)**0.5
                        dist_cost[r_i, c_i] = dist
                
                row_idx2, col_idx2 = linear_sum_assignment(dist_cost)
                
                for r_i, c_i in zip(row_idx2, col_idx2):
                    if dist_cost[r_i, c_i] <= 500.0:  # 500 pixels max distance
                        t_idx = unmatched_t_idx[r_i]
                        d_idx = unmatched_d_idx[c_i]
                        
                        self._tracks[t_idx].bbox = np.array(det_boxes[d_idx], dtype=np.float64)
                        
                        c_x = (det_boxes[d_idx][0] + det_boxes[d_idx][2]) / 2.0
                        c_y = (det_boxes[d_idx][1] + det_boxes[d_idx][3]) / 2.0
                        self._tracks[t_idx].history.append((c_x, c_y))
                        
                        self._tracks[t_idx].hits += 1
                        self._tracks[t_idx].age = 0
                        self._tracks[t_idx].confidence = float(confidences[d_idx])
                        
                        matched_trk.add(t_idx)
                        matched_det.add(d_idx)
            
            # Age unmatched tracks (after both passes)
            for t_idx in range(len(self._tracks)):
                trk = self._tracks[t_idx]
                if t_idx not in matched_trk:
                    trk.age += 1
                    # --- KCF TRACKING FALLBACK FOR LOST TRACKS ---
                    if frame is not None and trk.cv2_tracker is not None:
                        try:
                            ok, new_box = trk.cv2_tracker.update(frame)
                            if ok:
                                x, y, w, h = new_box
                                trk.bbox = np.array([x, y, x + w, y + h], dtype=np.float64)
                        except Exception:
                            pass
                else:
                    # Matched! Re-init tracker to sync with YOLO's new box
                    if frame is not None:
                        trk.cv2_tracker = _init_kcf(frame, trk.bbox)
                        trk.tracker_initialized = True

            # Create new tracks for unmatched detections
            for d_idx in unmatched_d_idx:
                c_x = (det_boxes[d_idx][0] + det_boxes[d_idx][2]) / 2.0
                c_y = (det_boxes[d_idx][1] + det_boxes[d_idx][3]) / 2.0
                new_trk = _Track(
                    track_id=self._alloc_id(),
                    bbox=np.array(det_boxes[d_idx], dtype=np.float64),
                    confidence=float(confidences[d_idx]),
                    history=[(c_x, c_y)]
                )
                if frame is not None:
                    new_trk.cv2_tracker = _init_kcf(frame, new_trk.bbox)
                    new_trk.tracker_initialized = True
                self._tracks.append(new_trk)

            # Remove dead tracks and save lost ids
            self._last_lost_ids = [t.track_id for t in self._tracks if t.age > self.max_age]
            self._tracks = [
                t for t in self._tracks if t.age <= self.max_age
            ]

        return [
            (t.track_id, t.bbox, t.confidence)
            for t in self._tracks
            if t.hits >= self.min_hits
        ]

    # ------------------------------------------------------------------
    def lost_ids(self) -> List[int]:
        """Return track IDs that just exceeded ``max_age`` (for EXIT events).

        Call *after* ``update()`` in the same frame.
        """
        return self._last_lost_ids

    # ------------------------------------------------------------------
    def _alloc_id(self) -> int:
        tid = SimpleTracker._next_id
        SimpleTracker._next_id += 1
        return tid


# ═══════════════════════════════════════════════════════════════════════════
# Point-in-polygon (ray-casting)
# ═══════════════════════════════════════════════════════════════════════════

def point_in_polygon(point: Point, polygon: Sequence[Point]) -> bool:
    """Classic ray-casting algorithm.

    Parameters
    ----------
    point:
        ``(x, y)`` to test.
    polygon:
        Ordered list of ``(x, y)`` vertices (automatically closed).

    Returns
    -------
    bool
    """
    x, y = point
    n = len(polygon)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / (yj - yi) + xi
        ):
            inside = not inside
        j = i
    return inside


# ═══════════════════════════════════════════════════════════════════════════
# High-level visitor tracker
# ═══════════════════════════════════════════════════════════════════════════

import os
import threading

# Module-level lock — works on all platforms (Windows, Linux, macOS).
# Prevents race conditions when multiple cameras generate visitor IDs in parallel.
_id_lock = threading.Lock()

def _make_visitor_id(track_id: int, camera_id: str) -> str:
    """Generate sequential unique IDs across all pipeline executions starting from 1000.
    Uses a threading lock to prevent race conditions during parallel processing."""
    counter_file = os.path.join(os.path.dirname(__file__), "..", "data", "id_counter.txt")
    with _id_lock:
        if not os.path.exists(counter_file):
            val = 1000
        else:
            try:
                with open(counter_file, "r") as f:
                    val = int(f.read().strip())
            except ValueError:
                val = 1000
        with open(counter_file, "w") as f:
            f.write(str(val + 1))
    return str(val)


@dataclass
class _VisitorState:
    """Mutable state for a single tracked visitor."""

    visitor_id: str
    track_id: int
    current_zone: Optional[str] = None
    zone_enter_time: Optional[datetime] = None
    last_dwell_emit: Optional[datetime] = None
    session_seq: int = 0
    is_staff: bool = False
    upper_hsv: Optional[tuple[int,int,int]] = None
    lower_hsv: Optional[tuple[int,int,int]] = None
    has_exited: bool = False
    bbox_history: List[Point] = field(default_factory=list)
    appearance_hist: Optional[np.ndarray] = None  # for cross-camera re-id
    confidence: float = 0.0
    is_probation: bool = True
    probation_events: List[Dict[str, Any]] = field(default_factory=list)
    spawn_centroid: Optional[Point] = None


class VisitorTracker:
    """High-level visitor tracking with zone awareness and event generation.

    Parameters
    ----------
    camera_id:
        The camera producing detections.
    zones:
        ``{zone_id: {"polygon": [(x,y), ...], "label": "..."}}``
    entry_threshold_y:
        Y-coordinate of the store entrance.  Movement *below* this line is
        "inbound"; movement *above* is "outbound".
    billing_zone_id:
        Zone id that represents the billing / checkout area.
    """

    _global_features: Dict[str, np.ndarray] = {}   # visitor_id → histogram
    _global_colors: Dict[str, Tuple[Tuple[int,int,int], Tuple[int,int,int]]] = {}  # visitor_id → (upper_bgr, lower_bgr)
    _global_id_map: Dict[str, str] = {}             # local_vid → canonical_vid
    _global_color_votes: Dict[str, Dict[tuple, float]] = {} # vid -> bin -> total_weight
    _staff_vids: set = set()
    _laptop_users: Dict[str, int] = {}
    _multiple_occurrence_colors: Dict[tuple, int] = {}
    _staff_color_hardcoded: tuple = None

    @classmethod
    def reset(cls):
        """Reset color voting states between passes, but KEEP Re-ID features
        so cross-camera matching works across clips."""
        cls._global_color_votes.clear()
        cls._staff_vids.clear()
        cls._laptop_users.clear()
        cls._multiple_occurrence_colors.clear()
        SimpleTracker._next_id = 0

    @classmethod
    def full_reset(cls):
        """Full reset including Re-ID features. Use between different stores."""
        cls._global_features.clear()
        cls._global_colors.clear()
        cls._global_id_map.clear()
        cls._global_color_votes.clear()
        cls._staff_vids.clear()
        cls._laptop_users.clear()
        cls._multiple_occurrence_colors.clear()
        SimpleTracker._next_id = 0

    @classmethod
    def update_staff_colors(cls):
        """Determine the staff color using Multi-Criteria Scoring."""
        if not cls._global_color_votes: return
        
        metrics = {} # bin -> {'multi': 0, 'votes': 0.0, 'laptop': 0}
        
        # 1. Total Votes
        for vid, votes in cls._global_color_votes.items():
            for bn, weight in votes.items():
                if bn not in metrics: metrics[bn] = {'multi': 0, 'votes': 0.0, 'laptop': 0}
                metrics[bn]['votes'] += weight
                
        # 2. Laptop Frames (assign to visitor's most common color)
        for vid, frames in cls._laptop_users.items():
            if vid in cls._global_color_votes and cls._global_color_votes[vid]:
                best_bin = max(cls._global_color_votes[vid].items(), key=lambda x: x[1])[0]
                if best_bin in metrics:
                    metrics[best_bin]['laptop'] += frames
                    
        # 3. Multiple Occurrences
        for bn, frames in cls._multiple_occurrence_colors.items():
            if bn in metrics:
                metrics[bn]['multi'] += frames
                
        if not metrics: return
        
        # Determine the winner based on priority: Multi -> Total Votes -> Laptop Interaction
        best_bin = max(metrics.keys(), key=lambda bn: (metrics[bn]['multi'], metrics[bn]['votes'], metrics[bn]['laptop']))
        cls._staff_color_hardcoded = best_bin
        
        # Print metrics for debugging
        print("\n--- MULTI-CRITERIA SCORING (TOP 5) ---")
        sorted_metrics = sorted(metrics.items(), key=lambda x: (x[1]['multi'], x[1]['votes'], x[1]['laptop']), reverse=True)
        for bn, m in sorted_metrics[:5]:
            print(f"Color {bn} -> Multi: {m['multi']}, Votes: {m['votes']:.1f}, Laptop: {m['laptop']}")
        print("--------------------------------------\n")
        
        # 4. For each visitor, check if any of their significant colors match the staff color
        cls._staff_vids = set()
        for vid, votes in cls._global_color_votes.items():
            if not votes: continue
            
            total_vid_votes = sum(votes.values())
            matching_votes = 0.0
            
            for bn, weight in votes.items():
                dists = [
                    abs(bn[0][0] - best_bin[0][0]),
                    abs(bn[0][1] - best_bin[0][1]),
                    abs(bn[0][2] - best_bin[0][2]),
                    abs(bn[1][0] - best_bin[1][0]),
                    abs(bn[1][1] - best_bin[1][1]),
                    abs(bn[1][2] - best_bin[1][2]),
                ]
                if all(d <= 1 for d in dists):
                    matching_votes += weight
            
            # Flag if uniform color was > 15% of their time or seen for > 15 frames
            if total_vid_votes > 0 and (matching_votes / total_vid_votes >= 0.15 or matching_votes >= 15.0):
                cls._staff_vids.add(vid)
                
        # 5. Also explicitly flag anyone who spent significant time operating the laptop
        for vid, frames in cls._laptop_users.items():
            if frames > 30: # More than 1 second interacting with laptop
                cls._staff_vids.add(vid)

    def __init__(
        self,
        camera_id: str,
        zones: Dict[str, Any],
        entry_threshold_y: Optional[float] = None,
        billing_zone_id: Optional[str] = None,
        is_billing_camera: bool = False,
    ) -> None:
        self.camera_id = camera_id
        self.zones = zones
        self.entry_threshold_y = entry_threshold_y
        self.billing_zone_id = billing_zone_id
        self.is_billing_camera = is_billing_camera

        self._tracker = SimpleTracker()
        self._visitors: Dict[int, _VisitorState] = {}  # track_id → state
        self._exited_visitors: Dict[int, _VisitorState] = {}
        self._billing_occupancy: int = 0
        
        # Deep Learning Re-ID Feature Extractor (MobileNetV3)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.reid_model = models.mobilenet_v3_small(pretrained=True).to(self.device)
        self.reid_model.eval()
        self.reid_transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def update(
        self,
        detections: np.ndarray,
        confidences: np.ndarray,
        frame: np.ndarray,
        frame_time: datetime,
        laptop_boxes: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """Process one frame's detections and return a list of event dicts.

        Parameters
        ----------
        detections:
            ``(N, 4)`` array of ``[x1, y1, x2, y2]``.
        confidences:
            ``(N,)`` array of detection confidences.
        frame:
            BGR image (used for staff classification).
        frame_time:
            Absolute timestamp of this frame.
        laptop_boxes:
            ``(M, 4)`` array of laptop/monitor bounding boxes detected
            in this frame. Used on billing cameras to identify the
            person standing behind the counter.

        Returns
        -------
        list[dict]
            Each dict has keys: ``event_type``, ``visitor_id``,
            ``timestamp``, ``zone_id``, ``dwell_ms``, ``is_staff``,
            ``confidence``, ``metadata``.
        """
        events: List[Dict[str, Any]] = []

        # Run low-level tracker.
        active = self._tracker.update(detections, confidences, frame)
        
        # Pre-compute: which person bbox overlaps most with a laptop/monitor?
        # Use IoU (Intersection over Union). A standing customer will have a massive bounding
        # box and thus a tiny IoU. A sitting employee will have a smaller bounding box and
        # a much higher IoU.
        self._laptop_nearest_tid = None
        if self.is_billing_camera and laptop_boxes is not None and len(laptop_boxes) > 0 and len(active) > 0:
            best_iou = 0.0
            for lap_box in laptop_boxes:
                lx1, ly1, lx2, ly2 = lap_box
                lap_area = (lx2 - lx1) * (ly2 - ly1)
                if lap_area <= 0: continue
                
                for tid, bbox, _ in active:
                    px1, py1, px2, py2 = bbox
                    
                    # Compute intersection
                    ix1 = max(lx1, px1)
                    iy1 = max(ly1, py1)
                    ix2 = min(lx2, px2)
                    iy2 = min(ly2, py2)
                    
                    if ix1 < ix2 and iy1 < iy2:
                        inter_area = (ix2 - ix1) * (iy2 - iy1)
                        person_area = (px2 - px1) * (py2 - py1)
                        union_area = lap_area + person_area - inter_area
                        iou = inter_area / union_area if union_area > 0 else 0
                        
                        if iou > best_iou:
                            best_iou = iou
                            self._laptop_nearest_tid = tid

        active_track_ids: set[int] = set()

        for track_id, bbox, conf in active:
            active_track_ids.add(track_id)

            # --- foot-point & centroid ---
            x1, y1, x2, y2 = bbox
            foot_point: Point = ((x1 + x2) / 2.0, y2)
            centroid: Point = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

            # --- get or create visitor state ---
            if track_id in self._visitors:
                vs = self._visitors[track_id]
                vs.confidence = conf
            elif track_id in self._exited_visitors:
                # Re-appeared after being lost → REENTRY
                vs = self._exited_visitors.pop(track_id)
                vs.has_exited = True
                vs.confidence = conf
                self._visitors[track_id] = vs
            else:
                vid = _make_visitor_id(track_id, self.camera_id)
                vs = _VisitorState(
                    visitor_id=vid,
                    track_id=track_id,
                    confidence=conf,
                    spawn_centroid=centroid,
                )
                # Dynamic color extraction
                vs.upper_hsv, vs.lower_hsv = self._extract_colors(frame, bbox)
                
                # Everyone starts as a general customer ID
                vs.visitor_id = f"C{vid}"
                    
                # Cross-camera re-id for everyone
                hist = self._compute_appearance_hist(frame, bbox)
                vs.appearance_hist = hist
                canonical = self._cross_camera_match(
                    vs.visitor_id, hist,
                    upper_bgr=vs.upper_hsv, lower_bgr=vs.lower_hsv,
                )
                if canonical is not None:
                    vs.visitor_id = canonical
                else:
                    # Register in global feature buffer + colors
                    if hist is not None:
                        VisitorTracker._global_features[vs.visitor_id] = hist
                        VisitorTracker._global_colors[vs.visitor_id] = (vs.upper_hsv, vs.lower_hsv)

                self._visitors[track_id] = vs

                # ENTRY or REENTRY
                evt_type = REENTRY if vs.has_exited else ENTRY
                vs.session_seq += 1
                direction = self._detect_direction(vs.bbox_history)
                
                new_evt = self._event_dict(
                    evt_type, vs, frame_time,
                    metadata={
                        "session_seq": vs.session_seq,
                        "direction": direction,
                    },
                )
                if vs.is_probation:
                    vs.probation_events.append(new_evt)
                else:
                    events.append(new_evt)

            # Update centroid history
            vs.bbox_history.append(centroid)
            if len(vs.bbox_history) > config.CENTROID_HISTORY_LEN:
                vs.bbox_history = vs.bbox_history[
                    -config.CENTROID_HISTORY_LEN:
                ]

        current_frame_bins = {}
        # Build O(1) lookup dict once per frame instead of scanning list per track
        track_dict = {t.track_id: t for t in self._tracker._tracks}

        for track_id in active_track_ids:
            vs = self._visitors[track_id]

            # O(1) lookup replacing the previous O(n^2) index+next scan
            bbox = track_dict[track_id].bbox
            
            # --- COLOR VOTING EVERY FRAME ---
            weight = 1.0
            
            if self._laptop_nearest_tid == track_id:
                VisitorTracker._laptop_users[vs.visitor_id] = VisitorTracker._laptop_users.get(vs.visitor_id, 0) + 1
            
            uc, lc = self._extract_colors(frame, bbox)
            def bin_color(c): return (c[0]//32, c[1]//32, c[2]//32)
            bn = (bin_color(uc), bin_color(lc))
            
            current_frame_bins[bn] = current_frame_bins.get(bn, 0) + 1
            
            if vs.visitor_id not in VisitorTracker._global_color_votes:
                VisitorTracker._global_color_votes[vs.visitor_id] = {}
            VisitorTracker._global_color_votes[vs.visitor_id][bn] = (
                VisitorTracker._global_color_votes[vs.visitor_id].get(bn, 0.0) + weight
            )

        # Update Multiple Occurrences
        for c_bin, count in current_frame_bins.items():
            if count >= 2:
                VisitorTracker._multiple_occurrence_colors[c_bin] = VisitorTracker._multiple_occurrence_colors.get(c_bin, 0) + 1
        
        for track_id in active_track_ids:
            vs = self._visitors[track_id]
            bbox = next(b for t, b, c in active if t == track_id)
            x1, y1, x2, y2 = bbox
            foot_point: Point = ((x1 + x2) / 2.0, y2)

            # --- zone checks ---
            new_zone = self._check_zone(foot_point)
            old_zone = vs.current_zone

            if new_zone != old_zone:
                # Left old zone?
                if old_zone is not None:
                    dwell = self._dwell_ms(vs.zone_enter_time, frame_time)
                    vs.session_seq += 1
                    evt1 = self._event_dict(
                        ZONE_EXIT, vs, frame_time,
                        zone_id=old_zone, dwell_ms=dwell,
                        metadata={
                            "session_seq": vs.session_seq,
                            "sku_zone": self._zone_label(old_zone),
                        },
                    )
                    if vs.is_probation: vs.probation_events.append(evt1)
                    else: events.append(evt1)
                    
                    # Billing zone leave
                    if old_zone == self.billing_zone_id:
                        self._billing_occupancy = max(
                            0, self._billing_occupancy - 1
                        )
                        vs.session_seq += 1
                        evt2 = self._event_dict(
                            BILLING_QUEUE_ABANDON, vs, frame_time,
                            zone_id=old_zone,
                            metadata={
                                "session_seq": vs.session_seq,
                                "queue_depth": self._billing_occupancy,
                            },
                        )
                        if vs.is_probation: vs.probation_events.append(evt2)
                        else: events.append(evt2)

                # Entered new zone?
                if new_zone is not None:
                    vs.zone_enter_time = frame_time
                    vs.last_dwell_emit = frame_time
                    vs.session_seq += 1
                    evt1 = self._event_dict(
                        ZONE_ENTER, vs, frame_time,
                        zone_id=new_zone,
                        metadata={
                            "session_seq": vs.session_seq,
                            "sku_zone": self._zone_label(new_zone),
                        },
                    )
                    if vs.is_probation: vs.probation_events.append(evt1)
                    else: events.append(evt1)
                    
                    # Billing zone join
                    if new_zone == self.billing_zone_id:
                        self._billing_occupancy += 1
                        vs.session_seq += 1
                        evt2 = self._event_dict(
                            BILLING_QUEUE_JOIN, vs, frame_time,
                            zone_id=new_zone,
                            metadata={
                                "session_seq": vs.session_seq,
                                "queue_depth": self._billing_occupancy,
                            },
                        )
                        if vs.is_probation: vs.probation_events.append(evt2)
                        else: events.append(evt2)

                vs.current_zone = new_zone

            else:
                # Still in same zone — emit periodic ZONE_DWELL
                if (
                    new_zone is not None
                    and vs.last_dwell_emit is not None
                ):
                    elapsed = (
                        frame_time - vs.last_dwell_emit
                    ).total_seconds() * 1000
                    if elapsed >= config.DWELL_INTERVAL_MS:
                        dwell = self._dwell_ms(
                            vs.zone_enter_time, frame_time
                        )
                        vs.session_seq += 1
                        evt1 = self._event_dict(
                            ZONE_DWELL, vs, frame_time,
                            zone_id=new_zone, dwell_ms=dwell,
                            metadata={
                                "session_seq": vs.session_seq,
                                "sku_zone": self._zone_label(new_zone),
                            },
                        )
                        if vs.is_probation: vs.probation_events.append(evt1)
                        else: events.append(evt1)
                        vs.last_dwell_emit = frame_time
            
            # Probation check (Proof of Life)
            if vs.is_probation and vs.spawn_centroid is not None:
                dx = centroid[0] - vs.spawn_centroid[0]
                dy = centroid[1] - vs.spawn_centroid[1]
                dist = (dx**2 + dy**2)**0.5
                if dist > 30.0:  # must travel 30 pixels from spawn to graduate
                    vs.is_probation = False
                    events.extend(vs.probation_events)
                    vs.probation_events.clear()

        # --- handle lost tracks → EXIT events ---
        lost_ids = set(self._visitors.keys()) - active_track_ids
        for tid in lost_ids:
            vs = self._visitors.pop(tid)
            
            # If track dies in probation, it was likely static noise/poster
            if vs.is_probation:
                continue

            # Zone exit if still in a zone
            if vs.current_zone is not None:
                dwell = self._dwell_ms(vs.zone_enter_time, frame_time)
                vs.session_seq += 1
                events.append(self._event_dict(
                    ZONE_EXIT, vs, frame_time,
                    zone_id=vs.current_zone, dwell_ms=dwell,
                    metadata={
                        "session_seq": vs.session_seq,
                        "sku_zone": self._zone_label(vs.current_zone),
                    },
                ))
                if vs.current_zone == self.billing_zone_id:
                    self._billing_occupancy = max(
                        0, self._billing_occupancy - 1
                    )

            vs.session_seq += 1
            direction = self._detect_direction(vs.bbox_history)
            events.append(self._event_dict(
                EXIT, vs, frame_time,
                metadata={
                    "session_seq": vs.session_seq,
                    "direction": direction,
                },
            ))
            vs.has_exited = True
            vs.current_zone = None
            self._exited_visitors[tid] = vs

        return events

    # ------------------------------------------------------------------

    def finalize(self, frame_time: datetime) -> List[Dict[str, Any]]:
        """Emit EXIT for every visitor still being tracked (end of clip)."""
        events: List[Dict[str, Any]] = []
        for tid in list(self._visitors.keys()):
            vs = self._visitors.pop(tid)
            if vs.current_zone is not None:
                dwell = self._dwell_ms(vs.zone_enter_time, frame_time)
                vs.session_seq += 1
                events.append(self._event_dict(
                    ZONE_EXIT, vs, frame_time,
                    zone_id=vs.current_zone, dwell_ms=dwell,
                    metadata={
                        "session_seq": vs.session_seq,
                        "sku_zone": self._zone_label(vs.current_zone),
                    },
                ))
            vs.session_seq += 1
            events.append(self._event_dict(
                EXIT, vs, frame_time,
                metadata={"session_seq": vs.session_seq, "reason": "clip_end"},
            ))
        return events

    # ------------------------------------------------------------------
    # Staff classification
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_colors(frame: np.ndarray, bbox: np.ndarray) -> tuple[tuple[int,int,int], tuple[int,int,int]]:
        """Extract dominant (Upper, Lower) BGR colors from the bounding box."""
        h, w = frame.shape[:2]
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(w, int(bbox[2]))
        y2 = min(h, int(bbox[3]))

        if x2 <= x1 or y2 <= y1:
            return ((0,0,0), (0,0,0))

        bh = y2 - y1
        bw = x2 - x1

        # Upper half
        uy1, uy2 = y1, y1 + bh//2
        # Lower half
        ly1, ly2 = y1 + bh//2, y2

        # Center 40% to avoid background
        cx1, cx2 = x1 + int(bw * 0.3), x2 - int(bw * 0.3)
        if cx2 <= cx1:
            cx1, cx2 = x1, x2 # Fallback if box is too thin

        upper_crop = frame[uy1:uy2, cx1:cx2]
        lower_crop = frame[ly1:ly2, cx1:cx2]

        def get_dom_bgr(crop):
            if crop.size == 0: return (0,0,0)
            # Simple BGR average is extremely fast and linear
            mean_c = cv2.mean(crop)[:3]
            return (int(mean_c[0]), int(mean_c[1]), int(mean_c[2]))

        return (get_dom_bgr(upper_crop), get_dom_bgr(lower_crop))

    # ------------------------------------------------------------------
    # Zone helpers
    # ------------------------------------------------------------------

    def _check_zone(self, foot_point: Point) -> Optional[str]:
        """Return the zone_id the *foot_point* falls inside, or ``None``."""
        for zone_id, zone_data in self.zones.items():
            polygon = zone_data.get("polygon", [])
            if point_in_polygon(foot_point, polygon):
                return zone_id
        return None

    def _zone_label(self, zone_id: Optional[str]) -> Optional[str]:
        """Human-readable label for a zone, if available."""
        if zone_id is None:
            return None
        return self.zones.get(zone_id, {}).get("label", zone_id)

    # ------------------------------------------------------------------
    # Direction detection
    # ------------------------------------------------------------------

    def _detect_direction(
        self,
        centroid_history: List[Point],
    ) -> Optional[str]:
        """Estimate movement direction relative to the entry threshold.

        Returns ``"inbound"`` (moving deeper into the store, i.e. increasing
        y), ``"outbound"`` (moving towards exit), or ``None`` when there
        is insufficient history or no threshold is configured.
        """
        if self.entry_threshold_y is None or len(centroid_history) < 2:
            return None
        first_y = centroid_history[0][1]
        last_y = centroid_history[-1][1]
        delta = last_y - first_y
        if abs(delta) < 5:
            return None
        return "inbound" if delta > 0 else "outbound"

    # ------------------------------------------------------------------
    # Cross-camera re-id
    # ------------------------------------------------------------------

    def _compute_appearance_hist(
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Compute a normalised HSV colour histogram of the upper body.

        This serves as a lightweight appearance feature for cross-camera
        matching — no deep re-id model required.
        """
        h, w = frame.shape[:2]
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(w, int(bbox[2]))
        y2 = min(h, int(bbox[3]))

        if x2 <= x1 or y2 <= y1:
            return None

        # Pass the full body crop for Deep Learning
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)
        tensor = self.reid_transform(pil_img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            feat = self.reid_model(tensor)
            
        feat = feat.squeeze().cpu().numpy()
        return feat

    @staticmethod
    def _color_distance(
        c1: Tuple[int, int, int],
        c2: Tuple[int, int, int],
    ) -> float:
        """Euclidean distance between two BGR colour tuples."""
        return float(((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2 + (c1[2]-c2[2])**2) ** 0.5)

    @classmethod
    def _cross_camera_match(
        cls,
        local_vid: str,
        hist: Optional[np.ndarray],
        upper_bgr: Optional[Tuple[int,int,int]] = None,
        lower_bgr: Optional[Tuple[int,int,int]] = None,
    ) -> Optional[str]:
        """Try to match *hist* against the global feature buffer.

        Uses a two-gate approach:
          1. Neural network cosine similarity (MobileNetV3 embeddings)
          2. Colour verification — upper and lower body BGR must be
             within ``COLOR_VERIFY_MAX_DIST`` Euclidean distance.

        Both gates must pass for a match to be accepted.  This prevents
        merging two different people who happen to produce similar generic
        'human' embeddings from the ImageNet-pretrained backbone.

        Returns the canonical ``visitor_id`` if a match is found, else
        ``None``.
        """
        COLOR_VERIFY_MAX_DIST = 80.0  # max BGR Euclidean dist per body half

        if hist is None or len(cls._global_features) == 0:
            return None

        best_vid: Optional[str] = None
        best_sim: float = -1.0

        for vid, stored_hist in cls._global_features.items():
            # Enforce Role Segregation: C only matches C, E only matches E
            if local_vid[0] != vid[0]:
                continue
            if vid == local_vid:
                continue
            # Cosine similarity
            dot = float(np.dot(hist, stored_hist))
            norm_a = float(np.linalg.norm(hist))
            norm_b = float(np.linalg.norm(stored_hist))
            if norm_a == 0 or norm_b == 0:
                continue
            sim = dot / (norm_a * norm_b)
            dist = 1.0 - sim
            if dist >= config.REID_DISTANCE_THRESHOLD:
                continue

            # ── Gate 2: Colour verification ──────────────────────────
            stored_colors = cls._global_colors.get(vid)
            if stored_colors is not None and upper_bgr is not None and lower_bgr is not None:
                s_upper, s_lower = stored_colors
                d_upper = cls._color_distance(upper_bgr, s_upper)
                d_lower = cls._color_distance(lower_bgr, s_lower)
                if d_upper > COLOR_VERIFY_MAX_DIST or d_lower > COLOR_VERIFY_MAX_DIST:
                    logger.debug(
                        "Re-ID colour gate blocked %s → %s  "
                        "(upper=%.0f, lower=%.0f, threshold=%.0f)",
                        local_vid, vid, d_upper, d_lower, COLOR_VERIFY_MAX_DIST,
                    )
                    continue  # colours too different — reject match

            if sim > best_sim:
                best_sim = sim
                best_vid = vid

        if best_vid is not None:
            cls._global_id_map[local_vid] = best_vid
            logger.info(
                "Cross-camera re-id: %s → %s (sim=%.3f)",
                local_vid,
                best_vid,
                best_sim,
            )
        return best_vid

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _dwell_ms(
        enter_time: Optional[datetime],
        now: datetime,
    ) -> int:
        """Milliseconds between *enter_time* and *now*."""
        if enter_time is None:
            return 0
        return max(0, int((now - enter_time).total_seconds() * 1000))

    @staticmethod
    def _event_dict(
        event_type: str,
        vs: _VisitorState,
        timestamp: datetime,
        zone_id: Optional[str] = None,
        dwell_ms: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build an event dict ready for ``EventEmitter.emit()``."""
        is_staff = vs.visitor_id in VisitorTracker._staff_vids

        # Dynamic color check: use the live dominant color bin accumulated via
        # voting (more robust than the spawn-time snapshot). Fall back to the
        # spawn-time upper_hsv/lower_hsv for visitors with no votes yet.
        if VisitorTracker._staff_color_hardcoded is not None:
            bn = VisitorTracker._staff_color_hardcoded

            # Try live dominant bin from color votes first
            vid_votes = VisitorTracker._global_color_votes.get(vs.visitor_id)
            if vid_votes:
                live_bin = max(vid_votes.items(), key=lambda x: x[1])[0]
                dists = [
                    abs(bn[0][0] - live_bin[0][0]),
                    abs(bn[0][1] - live_bin[0][1]),
                    abs(bn[0][2] - live_bin[0][2]),
                    abs(bn[1][0] - live_bin[1][0]),
                    abs(bn[1][1] - live_bin[1][1]),
                    abs(bn[1][2] - live_bin[1][2]),
                ]
            else:
                # Fallback: newly spawned visitor with no votes yet
                dists = [
                    abs(bn[0][0] - vs.upper_hsv[0]//32),
                    abs(bn[0][1] - vs.upper_hsv[1]//32),
                    abs(bn[0][2] - vs.upper_hsv[2]//32),
                    abs(bn[1][0] - vs.lower_hsv[0]//32),
                    abs(bn[1][1] - vs.lower_hsv[1]//32),
                    abs(bn[1][2] - vs.lower_hsv[2]//32),
                ]
            if all(d <= 1 for d in dists):
                is_staff = True
            
        return {
            "event_type": event_type,
            "visitor_id": vs.visitor_id,
            "timestamp": timestamp,
            "zone_id": zone_id,
            "dwell_ms": dwell_ms,
            "is_staff": is_staff,
            "confidence": vs.confidence,
            "metadata": metadata or {},
        }

    @property
    def billing_occupancy(self) -> int:
        """Current number of people in the billing zone."""
        return self._billing_occupancy
