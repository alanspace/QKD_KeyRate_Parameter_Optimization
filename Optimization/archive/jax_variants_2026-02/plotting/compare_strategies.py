import os
import json
import matplotlib.pyplot as plt
import numpy as np
import glob
from datetime import datetime

# Define strategies and their colors/styles
STRATEGIES = {
    "Baseline": {"color": "blue", "style": "--", "label": "Baseline (L-BFGS-B)"},
    "Hybrid": {"color": "green", "style": "-", "label": "Hybrid (L-BFGS + DA)"},
    "Pure_DA": {"color": "red", "style": "-.", "label": "Pure Dual Annealing"},
    "Ultra": {"color": "purple", "style": ":", "label": "Ultra Precision"},
    "Hessian": {"color": "orange", "style": "--", "label": "Hessian (Trust-NCG)"},
    "Hessian_FoR": {"color": "red", "style": "-", "label": "FoR Hessian + Warming"}
}

OUTPUT_DIR = "Comparison"

def find_latest_run(strategy_name):
    """Finds the latest run folder for a given strategy."""
    search_path = os.path.join(".", strategy_name, "run_*")
    runs = glob.glob(search_path)
    if not runs:
        return None
    # Sort by name (timestamp)
    return sorted(runs)[-1]

def load_run_data(run_folder):
    """Loads the grouped dataset from a run folder."""
    json_files = glob.glob(os.path.join(run_folder, "qkd_grouped_dataset_*.json"))
    if not json_files:
        return None
    
    with open(json_files[0], 'r') as f:
        return json.load(f)

def plot_comparisons():
    print("🔍 Scanning for latest runs...")
    
    plt.figure(figsize=(12, 8))
    
    data_found = False
    
    # We will plot for a specific n_X, let's say 1e9, or all?
    # Let's plot for n_X = 1e9 (most critical) and maybe 1e5 for short distance check.
    # For clarity, let's stick to the highest n_X available in all.
    
    target_nX = 1e9 # Default high precision target
    
    for strat, style in STRATEGIES.items():
        latest_run = find_latest_run(strat)
        if not latest_run:
            print(f"⚠️  No runs found for {strat}")
            continue
            
        print(f"✅ Found {strat}: {latest_run}")
        data = load_run_data(latest_run)
        if not data:
            print(f"   ❌ No data file in {latest_run}")
            continue

        # Find data for target n_X
        # Data keys are strings of floats, e.g. "1000000000.0"
        available_keys = list(data.keys())
        # Try to find 1e9, else max
        key_to_plot = str(float(target_nX))
        if key_to_plot not in data:
            # Fallback to max n_X
            floats = [float(k) for k in available_keys]
            if not floats: continue
            max_nx = max(floats)
            key_to_plot = str(float(max_nx))
            print(f"   ℹ️  n_X={target_nX} not found, using {key_to_plot}")
        
        subset = data[key_to_plot]
        # Sort by length
        subset.sort(key=lambda x: x["fiber_length"])
        
        lengths = [x["fiber_length"] for x in subset]
        rates = [x["key_rate"] for x in subset]
        
        # Filter zero rates for log plot
        clean_len = []
        clean_rate = []
        for l, r in zip(lengths, rates):
            if r > 1e-20:
                clean_len.append(l)
                clean_rate.append(r)
        
        if clean_len:
            plt.plot(clean_len, np.log10(clean_rate), 
                     color=style["color"], 
                     linestyle=style["style"], 
                     label=f"{style['label']} (n_X={float(key_to_plot):.0e})",
                     linewidth=2)
            data_found = True

    if not data_found:
        print("❌ No valid data found to plot.")
        return

    plt.xlabel("Fiber Length (km)", fontsize=14)
    plt.ylabel("log10(Key Rate)", fontsize=14)
    plt.title("Comparison of Optimization Strategies (Latest Runs)", fontsize=16)
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend(fontsize=12)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(OUTPUT_DIR, f"comparison_latest_{timestamp}.png")
    plt.savefig(output_path, dpi=300)
    print(f"\n📊 Comparison plot saved to: {output_path}")

if __name__ == "__main__":
    plot_comparisons()
