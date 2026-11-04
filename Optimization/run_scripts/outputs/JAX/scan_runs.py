import os
import json
import glob
from datetime import datetime

def load_metadata(run_folder):
    """Loads metadata from run_metadata.json if it exists."""
    meta_path = os.path.join(run_folder, "run_metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as f:
                return json.load(f)
        except:
            return None
    return None

def scan_runs():
    base_dir = "."
    # Define subdirectories to scan
    subdirs = [".", "Baseline", "Hybrid", "Pure_DA", "Ultra", "Hessian", "Hessian_FoR"]
    
    runs = []

    for subdir in subdirs:
        if not os.path.exists(subdir): continue
        
        # Find all run folders in this subdirectory
        for item in os.listdir(subdir):
            full_path = os.path.join(subdir, item)
            if os.path.isdir(full_path) and item.startswith("run_"):
                # Load Metadata
                metadata = load_metadata(full_path)
                
                # Default values if no metadata
                script_name = "Unknown Script"
                description = "Legacy Run (No Metadata)"
                strategy = "Unknown"
                
                if metadata:
                    script_name = metadata.get("script_name", "Unknown")
                    description = metadata.get("description", "")
                    strategy = metadata.get("strategy", "Unknown")
                else:
                    # Fallback Logic for Legacy Runs
                    if "BASELINE" in item: strategy = "Baseline"
                    elif "HYBRID" in item: strategy = "Hybrid"
                    elif "PURE_DA" in item: strategy = "Pure DA"
                    elif "ULTRA" in item: strategy = "Ultra"
                
                # Parse Timestamp from Folder Name
                try:
                    parts = item.split("_")
                    timestamp_str = f"{parts[-2]}_{parts[-1]}"
                    dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    time_display = dt.strftime("%Y-%m-%d %H:%M")
                except:
                    time_display = "Invalid Time"

                # Check for Results
                json_files = glob.glob(os.path.join(full_path, "qkd_grouped_dataset_*.json"))
                status = f"✅ {len(json_files)} Data" if json_files else "⚠️ Empty"
                
                runs.append({
                    "folder": item,
                    "path": full_path,
                    "strategy": strategy,
                    "time": time_display,
                    "status": status,
                    "script": script_name,
                    "desc": description
                })

    # Sort runs by timestamp (newest first)
    runs.sort(key=lambda x: x["time"], reverse=True)
    
    # Display Table
    print(f"{'Folder Name':<35} | {'Strategy':<20} | {'Time':<16} | {'Script Used':<30} | {'Status'}")
    print("-" * 120)
    
    for r in runs:
        # truncate script name if too long
        s_name = (r['script'][:27] + '..') if len(r['script']) > 29 else r['script']
        print(f"{r['folder']:<35} | {r['strategy']:<20} | {r['time']:<16} | {s_name:<30} | {r['status']}")

if __name__ == "__main__":
    scan_runs()
