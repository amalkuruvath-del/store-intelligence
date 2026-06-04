import os
import glob
import argparse
from visualize import run_visualizer

parser = argparse.ArgumentParser()
parser.add_argument("--store", default="Store 1", help="Store name (e.g. 'Store 1')")
parser.add_argument("--store-id", default="STORE_01", help="Store ID (e.g. 'STORE_01')")
args = parser.parse_args()

data_dir = rf"..\dataset\{args.store}"
layout_path = os.path.join(data_dir, "store_layout.json")

extensions = ("*.mp4", "*.avi", "*.mkv", "*.mov")
clips = []
for ext in extensions:
    clips.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
clips = sorted(clips)

print(f"Found {len(clips)} clip(s) to process in {args.store}.")

# Create output dir if not exists
os.makedirs("output", exist_ok=True)

for vid_path in clips:
    filename = os.path.basename(vid_path)
    name_no_ext = os.path.splitext(filename)[0]
    
    if "__" in name_no_ext:
        cam_id = name_no_ext.split("__")[0]
    else:
        cam_id = name_no_ext

    store_name_nospace = args.store.replace(" ", "")
    out_name = os.path.join("output", f"{store_name_nospace}_{name_no_ext}_debug.mp4")
    
    print(f"\nStarting visualizer for {cam_id} ({filename})...")
    run_visualizer(
        video_path=vid_path,
        store_id=args.store_id,
        camera_id=cam_id,
        layout_path=layout_path,
        out_path=out_name
    )
    print(f"Finished {cam_id}")

print("All done!")
