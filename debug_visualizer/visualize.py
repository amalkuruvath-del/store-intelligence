import sys
import os
import cv2
import numpy as np
from ultralytics import YOLO
import argparse
from datetime import datetime, timedelta

# Add parent dir to path to import pipeline
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline import config
from pipeline.tracker import VisitorTracker, point_in_polygon

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


def run_visualizer(video_path, store_id, camera_id, layout_path, out_path):
    import json
    with open(layout_path, "r") as f:
        layout = json.load(f)
        
    camera_cfg = layout.get("cameras", {}).get(camera_id, {})
    zones = camera_cfg.get("zones", [])
    zone_dict = {
        z_id: layout["zones"][z_id]
        for z_id in zones
        if z_id in layout.get("zones", {})
    }
    
    tracker = VisitorTracker(
        camera_id=camera_id,
        zones=zone_dict,
        entry_threshold_y=layout.get("entry_threshold_y"),
        billing_zone_id=layout.get("billing_zone_id"),
    )
    
    model = YOLO(os.path.join('..', config.MODEL_PATH))
    
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0: video_fps = 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, video_fps, (width, height))
    
    start_time = datetime.now()
    
    # --- PASS 1: Tracking & Color Voting (No Rendering) ---
    print(f"Starting Pass 1 (Tracking): {video_path}")
    frame_data_buffer = [] # Store track info for each frame
    frame_count = 0
    
    is_billing_camera = False
    billing_zone_id = layout.get("billing_zone_id")
    if camera_cfg and billing_zone_id in camera_cfg.get("zones", []):
        is_billing_camera = True
        
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # YOLO inference - Class 0 (Person), 62 (TV), 63 (Laptop)
        results = model(frame, conf=0.65, iou=config.IOU_THRESHOLD, classes=[0, 62, 63], stream=True, verbose=False)
        det_list = []
        conf_list = []
        lap_list = []
        
        roi = camera_cfg.get("roi") or layout.get("roi")
        frame_area = width * height
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if conf < 0.65: # Hardcoded high threshold to remove empty ghost boxes
                    continue
                    
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                if cls_id == 62 or cls_id == 63:
                    lap_list.append([x1, y1, x2, y2])
                    continue
                    
                if cls_id != 0:
                    continue
                
                w = x2 - x1
                h = y2 - y1
                
                if h > 0 and (w / h) > 1.2:
                    continue
                if (w * h) > (0.4 * frame_area):
                    continue
                    
                foot = ((x1 + x2) / 2.0, y2)
                if roi and not point_in_polygon(foot, roi):
                    continue
                    
                det_list.append([x1, y1, x2, y2])
                conf_list.append(conf)
                
        detections = np.array(det_list) if det_list else np.empty((0, 4))
        confidences = np.array(conf_list) if conf_list else np.empty((0,))
        laptop_boxes = np.array(lap_list) if lap_list else np.empty((0, 4))
        
        if len(detections) > 0:
            detections, confidences = apply_iom_nms(detections, confidences, iom_threshold=0.8)
        
        # Update Tracker
        frame_time = start_time + timedelta(seconds=frame_count / video_fps)
        tracker.update(detections, confidences, frame, frame_time, laptop_boxes=laptop_boxes)
        
        # Save frame data for Pass 2
        current_frame_tracks = []
        for tid, vs in tracker._visitors.items():
            trk = next((t for t in tracker._tracker._tracks if t.track_id == tid), None)
            if trk:
                current_frame_tracks.append({
                    "bbox": trk.bbox,
                    "visitor_id": vs.visitor_id,
                    "is_probation": vs.is_probation,
                    "confidence": vs.confidence,
                    "upper_hsv": vs.upper_hsv,
                    "lower_hsv": vs.lower_hsv
                })
        frame_data_buffer.append({
            "tracks": current_frame_tracks,
            "laptops": laptop_boxes
        })
        
        frame_count += 1
        if frame_count % 100 == 0: print(f"Pass 1: Processed {frame_count} frames for {camera_id}")

    # --- FINAL STAFF CHECKUP & DEBUG ---
    print("\n================ COLOR VOTING DEBUG ================")
    print("--- LAPTOP INTERACTION COUNTS ---")
    for vid, count in sorted(VisitorTracker._laptop_users.items(), key=lambda x: x[1], reverse=True):
        print(f"Visitor {vid} -> Interacted with laptop for {count} frames")
        
    print("\n--- RAW COLOR VOTES ---")
    # Print the raw color votes per visitor
    for vid, votes in VisitorTracker._global_color_votes.items():
        if not votes: continue
        top_color = max(votes.items(), key=lambda x: x[1])
        total_votes = sum(votes.values())
        print(f"Visitor {vid} -> Total Votes: {total_votes:.1f} | Top Color: {top_color[0]} ({top_color[1]:.1f} votes)")
    
    # Run the actual clustering update
    print("\nRunning final color voting to determine exact Employee Color...")
    VisitorTracker.update_staff_colors()
    if hasattr(VisitorTracker, '_staff_color_hardcoded'):
        sc = VisitorTracker._staff_color_hardcoded
        print(f"Definitive Staff Color Code (Upper/Lower): {sc}")
    print("====================================================\n")
    
    # --- PASS 2: Rendering ---
    print(f"Starting Pass 2 (Rendering) to {out_path}...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Rewind video
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= len(frame_data_buffer): break
        
        frame_data = frame_data_buffer[frame_idx]
        tracks = frame_data["tracks"]
        laptops = frame_data["laptops"]
        
        # Draw laptops
        for lx1, ly1, lx2, ly2 in laptops:
            cv2.rectangle(frame, (int(lx1), int(ly1)), (int(lx2), int(ly2)), (255, 0, 0), 2)
            cv2.putText(frame, "LAPTOP", (int(lx1), int(ly1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        for t in tracks:
            x1, y1, x2, y2 = t["bbox"]
            is_staff = False
            if hasattr(VisitorTracker, '_staff_color_hardcoded') and VisitorTracker._staff_color_hardcoded:
                bn = VisitorTracker._staff_color_hardcoded
                uh = t["upper_hsv"]
                lh = t["lower_hsv"]
                dists = [
                    abs(bn[0][0] - uh[0]//32),
                    abs(bn[0][1] - uh[1]//32),
                    abs(bn[0][2] - uh[2]//32),
                    abs(bn[1][0] - lh[0]//32),
                    abs(bn[1][1] - lh[1]//32),
                    abs(bn[1][2] - lh[2]//32),
                ]
                if all(d <= 1 for d in dists):
                    is_staff = True
            
            if t["is_probation"]:
                color = (0, 255, 255)
                label = "Probation"
                thickness = 2
            elif is_staff:
                color = (0, 0, 255)
                label = "E" + t["visitor_id"][1:]
                thickness = 3
            else:
                color = (0, 255, 0)
                label = t["visitor_id"]
                thickness = 3
                
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
            cv2.putText(frame, f"{label} ({t['confidence']:.2f})", (int(x1), int(y1)-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f"{label} ({t['confidence']:.2f})", (int(x1), int(y2)+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw Zones
        for zid, zd in zone_dict.items():
            poly = np.array(zd['polygon'], dtype=np.int32)
            cv2.polylines(frame, [poly], True, (255, 255, 0), 2)
            if len(poly) > 0:
                cv2.putText(frame, zd.get('label', zid), tuple(poly[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        out.write(frame)
        frame_idx += 1
        if frame_idx % 100 == 0: print(f"Pass 2: Rendered {frame_idx} frames for {camera_id}")

    cap.release()
    out.release()
    print("Saved to", out_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--store-id", default="ST1008")
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--layout", default="..\\data\\store_layout.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    
    run_visualizer(args.video, args.store_id, args.camera_id, args.layout, args.out)
