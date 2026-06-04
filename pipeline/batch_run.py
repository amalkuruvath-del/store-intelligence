import os
import sys
import glob
import logging
from datetime import datetime, timezone
import subprocess

from pipeline.detect import run, _parse_start_time
from pipeline.jsonl_export import regenerate_jsonl

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--store-id", default="STORE_01")
    parser.add_argument("--layout", default="store_layout.json")
    args = parser.parse_args()

    data_dir = args.data_dir
    store_id = args.store_id
    layout_path = args.layout

    if not os.path.isdir(data_dir):
        print(f"[ERROR] Data directory not found: {data_dir}")
        sys.exit(1)

    # If layout is a relative default, resolve it inside data_dir
    if not os.path.isabs(layout_path) and not os.path.isfile(layout_path):
        candidate = os.path.join(data_dir, layout_path)
        if os.path.isfile(candidate):
            layout_path = candidate

    if not os.path.isfile(layout_path):
        print(f"[FATAL ERROR] Layout file '{layout_path}' not found!")
        print("Please ensure 'store_layout.json' is present in the specified directory.")
        sys.exit(1)

    # Find all clips
    extensions = ("*.mp4", "*.avi", "*.mkv", "*.mov")
    clips = []
    for ext in extensions:
        clips.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
    
    clips = sorted(clips)
    if not clips:
        print(f"[ERROR] No video clips found in {data_dir}")
        sys.exit(1)
    
    print(f"[INFO]  Found {len(clips)} clip(s) in {data_dir}")

    processed = 0
    failed = 0

    for clip in clips:
        filename = os.path.basename(clip)
        name_no_ext = os.path.splitext(filename)[0]
        
        if "__" in name_no_ext:
            camera_id = name_no_ext.split("__")[0]
        else:
            camera_id = name_no_ext
        
        # Determine timestamp
        import re
        match = re.search(r'([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}-[0-9]{2}-[0-9]{2})', name_no_ext)
        if match:
            raw_ts = match.group(1)
            start_time_str = f"{raw_ts[0:11]}{raw_ts[11:13]}:{raw_ts[14:16]}:{raw_ts[17:19]}Z"
        else:
            # use mtime
            mtime = os.path.getmtime(clip)
            dt = datetime.fromtimestamp(mtime, tz=timezone.utc)
            start_time_str = dt.isoformat()

        print("----------------------------------------------------")
        print(f"[INFO]  Processing: {clip}")
        print(f"[INFO]    Store:    {store_id}")
        print(f"[INFO]    Camera:   {camera_id}")
        print(f"[INFO]    Start:    {start_time_str}")
        print(f"[INFO]    Layout:   {layout_path}")
        print("----------------------------------------------------")

        start_time = _parse_start_time(start_time_str)

        try:
            run(
                video_path=clip,
                store_id=store_id,
                camera_id=camera_id,
                layout_path=layout_path,
                start_time=start_time
            )
            processed += 1
        except Exception as e:
            print(f"[ERROR] Failed to process {clip}: {e}")
            failed += 1

        # Regenerate filtered JSONL after every clip so the log file
        # is always in a clean state, even if the pipeline is killed.
        regenerate_jsonl(store_id)

    print("\n=======================================================")
    print("               FINAL GLOBAL STAFF CHECKUP")
    print("=======================================================")
    from pipeline.tracker import VisitorTracker
    VisitorTracker.update_staff_colors()
    staff_vids = list(VisitorTracker._staff_vids)
    if staff_vids:
        staff_ids_str = ", ".join(f"'{vid}'" for vid in staff_vids)
        sql = f"UPDATE events SET is_staff = CASE WHEN visitor_id IN ({staff_ids_str}) THEN TRUE ELSE FALSE END WHERE store_id = '{store_id}';"
    else:
        sql = f"UPDATE events SET is_staff = FALSE WHERE store_id = '{store_id}';"
        
    try:
        subprocess.run([
            "docker", "exec", "purple-db-1", "psql", "-U", "postgres", "-d", "store_intelligence", "-c", sql
        ], check=True)
        print(f"[INFO] Updated {len(staff_vids)} staff IDs globally in the database.")
    except Exception as e:
        print(f"[ERROR] Failed to run global staff update: {e}")

    # Final JSONL regeneration with perfectly corrected staff flags
    print("[INFO] Regenerating event_log.jsonl with final staff corrections...")
    final_count = regenerate_jsonl(store_id)
    print(f"[INFO] Final event_log.jsonl contains {final_count} filtered events for {store_id}")

    print("\n=======================================================")
    print("               PIPELINE RUN COMPLETE")
    print("=======================================================")
    print(f"[INFO]    Total clips:     {len(clips)}")
    print(f"[INFO]    Processed OK:    {processed}")
    print(f"[INFO]    Failed:          {failed}")
    print("=======================================================")
    
    sys.exit(failed)

if __name__ == "__main__":
    main()
