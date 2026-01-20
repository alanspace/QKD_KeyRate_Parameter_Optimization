
import numpy as np
import matplotlib.pyplot as plt
import json
import glob
import sys
import os

# Ensure we can import from src
# Assuming this script is run from project root or handles paths correctly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Handle JAX Configuration - FORCE CPU to avoid Metal issues for simple plotting
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import jax
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)

# Try imports
try:
    from src.qkd.model import calculate_key_rates_and_metrics
    print("✅ Imported physics model successfully.")
except ImportError:
    # Fallback if running from root directly
    if os.getcwd() not in sys.path:
        sys.path.append(os.getcwd())
    from src.qkd.model import calculate_key_rates_and_metrics
    print("✅ Imported physics model successfully (fallback).")

# Enable LaTeX rendering for text (using MathText, not full LaTeX for speed/compat)
plt.rc('text', usetex=False)

def get_latest_dataset():
    # Search in proper location
    search_path = os.path.join(project_root, "Training_Data/n_X/good/reordered_qkd_grouped_dataset_*.json")
    files = sorted(glob.glob(search_path), reverse=True)
    if not files:
        # Fallback search path
        search_path = "Training_Data/n_X/good/reordered_qkd_grouped_dataset_*.json"
        files = sorted(glob.glob(search_path), reverse=True)
    
    if files:
        print(f"🔗 Loaded dataset: {files[0]}")
        with open(files[0], 'r') as f:
            return json.load(f)
    else:
        print("❌ Error: No dataset found.")
        return None

def save_dynamic_vs_static_overlay(dataset):
    print("\n--- Generating: Dynamic vs Static Overlay ---")
    
    # 1. Get Optimized Data (Blue) for n_X = 1e8
    target_nx = "100000000.0"
    if dataset and target_nx in dataset:
        opt_data = [d for d in dataset[target_nx] if d['key_rate'] > 0]
        opt_L = [d['fiber_length'] for d in opt_data]
        opt_R = [d['key_rate'] for d in opt_data]
    else:
        print(f"⚠️ Warning: n_X={target_nx} not found in dataset. Optimzed curve will be empty.")
        opt_L, opt_R = [], []

    # 2. Calculate Static Data (Red)
    # Parameters (Compromise set)
    params_tuple = (0.34479867337905723, 0.19526100517866424, 0.21504895346866, 0.4645950203307233, 0.1)
    n_X = 1e8
    alpha = 0.2
    eta_Bob = 0.1
    P_dc_value = 6e-7
    epsilon_sec = 1e-10
    epsilon_cor = 1e-15
    f_EC = 1.16
    e_mis = 0.01
    P_ap = 0
    n_event = 1

    L_values = np.linspace(0, 200, 200)
    static_key_rates = []

    print("Calculating static curve...")
    for L in L_values:
        metrics = calculate_key_rates_and_metrics(params_tuple, L, n_X, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event)
        rate = float(metrics[0])
        if rate < 0: rate = 0.0
        static_key_rates.append(rate)

    # 3. Plot
    plt.figure(figsize=(10, 6))
    if opt_L:
        plt.semilogy(opt_L, opt_R, 'b-', linewidth=2, label='Optimized (Dynamic)')
    plt.semilogy(L_values, static_key_rates, 'r--', linewidth=2, label='Static (Fixed Params)')

    plt.xlabel('Distance (km)')
    plt.ylabel('Secret Key Rate (log)')
    plt.title(f'True Comparison: Dynamic vs Static (n_X={n_X:.0e})')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.4)
    plt.xlim(0, 200)
    plt.ylim(1e-9, 1e-1)
    
    # Save
    out_dir = os.path.join(project_root, "Testing")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f"Created directory: {out_dir}")
        
    out_path = os.path.join(out_dir, "Dynamic_vs_Static_Overlay.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved overlay plot to: {out_path}")
    plt.close()

def save_key_rate_summary(dataset):
    print("\n--- Generating: Key Rate Summary Plot ---")
    if not dataset:
        return

    n_X_values = [10**s for s in range(4, 10)]
    plt.figure(figsize=(10, 6))

    for n_X in n_X_values:
        target_nx_str = str(float(n_X))
        if target_nx_str not in dataset:
            continue

        filtered_data = dataset[target_nx_str]
        # Filter noise
        filtered_data = [d for d in filtered_data if d["key_rate"] > 1e-20]
        
        if not filtered_data:
            continue

        fiber_lengths = [d["fiber_length"] for d in filtered_data]
        key_rates = [d["key_rate"] for d in filtered_data]
        
        exponent = int(np.log10(n_X))
        plt.plot(fiber_lengths, np.log10(key_rates), linestyle='-', label=r'$n_X = 10^{{{}}}$'.format(exponent))

    plt.xlabel("Fiber Length (km)")
    plt.ylabel("Secret Key Rate per Pulse")
    plt.xlim(0, 200)
    plt.ylim(-10, -1.5)
    plt.yticks(np.arange(-10, -1), [f"$10^{{{i}}}$" for i in range(-10, -1)])
    plt.title("Key Rate vs Fiber Length for Different $n_X$ Values")
    plt.legend()
    plt.grid(True)
    
    # Save
    out_dir = os.path.join(project_root, "Training_Data/n_X/good")
    # Dir should exist but verify
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    out_path = os.path.join(out_dir, "key_rate_vs_fiber_length.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"✅ Saved summary plot to: {out_path}")
    plt.close()

def main():
    dataset = get_latest_dataset()
    if dataset:
        save_dynamic_vs_static_overlay(dataset)
        save_key_rate_summary(dataset)
    else:
        print("Skipping plots due to missing dataset.")

if __name__ == "__main__":
    main()
