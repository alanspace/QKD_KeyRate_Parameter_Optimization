import json
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import argparse

def load_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_comparison(file1, file2, output_dir="."):
    data1 = load_data(file1)
    data2 = load_data(file2)
    
    name1 = os.path.basename(os.path.dirname(file1)) # e.g. run_2026...
    name2 = os.path.basename(os.path.dirname(file2))
    
    # intersection of nx keys
    keys1 = set(data1.keys())
    keys2 = set(data2.keys())
    common_keys = keys1.intersection(keys2)
    
    if not common_keys:
        print("No common n_X values found between the two datasets.")
        return

    sorted_keys = sorted(list(common_keys), key=lambda x: float(x))
    
    for nx in sorted_keys:
        entries1 = data1[nx]
        entries2 = data2[nx]
        
        # Sort by length
        entries1.sort(key=lambda x: x['fiber_length'])
        entries2.sort(key=lambda x: x['fiber_length'])
        
        L1 = [e['fiber_length'] for e in entries1 if e['key_rate'] > 1e-20]
        R1 = [e['key_rate'] for e in entries1 if e['key_rate'] > 1e-20]
        
        L2 = [e['fiber_length'] for e in entries2 if e['key_rate'] > 1e-20]
        R2 = [e['key_rate'] for e in entries2 if e['key_rate'] > 1e-20]
        
        plt.figure(figsize=(10, 6))
        
        if L1: plt.plot(L1, np.log10(R1), 'b-', label=f"Run 1: {name1}", linewidth=2)
        if L2: plt.plot(L2, np.log10(R2), 'r--', label=f"Run 2: {name2}", linewidth=2)
        
        plt.xlabel("Fiber Length (km)")
        plt.ylabel("log10(Key Rate)")
        plt.title(f"Comparison: n_X = {float(nx):.0e}")
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.legend()
        
        outfile = os.path.join(output_dir, f"compare_nx_{float(nx):.0e}.png")
        plt.savefig(outfile, dpi=300)
        plt.close()
        print(f"Saved comparison plot: {outfile}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare two QKD optimization runs.")
    parser.add_argument("file1", help="Path to first qkd_grouped_dataset_*.json")
    parser.add_argument("file2", help="Path to second qkd_grouped_dataset_*.json")
    parser.add_argument("--out", help="Output directory", default=".")
    
    args = parser.parse_args()
    
    plot_comparison(args.file1, args.file2, args.out)
