import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

# Setup Path to import src
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.qkd.model import scalar_objective

# Constants (Must match Notebook)
alpha = 0.2
eta_Bob = 0.1
P_dc_value = 6e-7
epsilon_sec = 1e-10
epsilon_cor = 1e-15
f_EC = 1.16
e_mis = 0.01
P_ap = 0
n_event = 1

constants = (alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event)

# Fixed Parameters Map (Same as used in script optimization)
INITIAL_GUESSES_MAP = {
    1e4: np.array([0.65, 0.15, 0.05, 0.61, 0.425]),
    1e5: np.array([0.62, 0.24, 0.10, 0.70, 0.55]),
    1e6: np.array([0.68, 0.30, 0.14, 0.74, 0.66]),
    1e7: np.array([0.55, 0.34, 0.15, 0.75, 0.75]),
    1e8: np.array([0.54, 0.375, 0.16, 0.775, 0.83]),
    1e9: np.array([0.52, 0.40, 0.18, 0.785, 0.88]),
}
DEFAULT_GUESS = np.array([0.52, 0.40, 0.18, 0.785, 0.88])

def calculate_fixed_curve(L_values, n_X, fixed_params):
    rates = []
    p_jax = jnp.array(fixed_params)
    for L in L_values:
        try:
            r = scalar_objective(p_jax, float(L), float(n_X), *constants)
            # scalar_objective returns negative rate
            rates.append(-float(r))
        except Exception as e:
            rates.append(0.0)
    return np.array(rates)

def main():
    # Path to the specific notebook result file
    # /Users/daai6ga1hou2/Documents/GitHub/Physics/QKD_KeyRate_Parameter_Optimization/Training_Data/n_X/good/reordered_qkd_grouped_dataset_20260204_172629.json
    target_file = os.path.join(project_root, "Training_Data", "n_X", "good", "reordered_qkd_grouped_dataset_20260204_172629.json")
    
    if not os.path.exists(target_file):
        print(f"Error: File {target_file} not found.")
        return

    print(f"Loading notebook results from {target_file}")
    with open(target_file, 'r') as f:
        data = json.load(f)

    # Prepare Plot
    manual_colors = ['red', 'green', 'blue', 'purple', 'cyan', 'orange']
    import itertools
    
    nx_keys = sorted(data.keys(), key=lambda x: float(x))
    line_colors = list(itertools.islice(itertools.cycle(manual_colors), len(nx_keys)))
    
    plt.figure(figsize=(10, 8))
    print("Generating Comparison Plot...")

    for i, nx_str in enumerate(nx_keys):
        n_X = float(nx_str)
        entries = data[nx_str]
        
        # Sort by fiber length
        entries.sort(key=lambda x: x["fiber_length"])
        
        L = np.array([e["fiber_length"] for e in entries])
        # Note: key "key_rate" might be directly accessible
        R_opt = np.array([e["key_rate"] for e in entries])
        
        # Calculate Fixed Rate
        fixed_params = INITIAL_GUESSES_MAP.get(n_X, DEFAULT_GUESS)
        R_fixed = calculate_fixed_curve(L, n_X, fixed_params)
        
        color = line_colors[i]
        
        # Plot Optimized (Notebook Result)
        plt.semilogy(L, R_opt, '-', color=color, linewidth=2, label=f'Notebook Opt $n_X=10^{int(np.log10(n_X))}$')
        
        # Plot Fixed Ref
        plt.semilogy(L, R_fixed, '--', color=color, linewidth=1.5, alpha=0.7)
        # Dummy legend for Fixed
        plt.semilogy([], [], '--', color=color, label=f'Fixed $n_X=10^{int(np.log10(n_X))}$')

    plt.title("Notebook Optimized vs. Fixed-Parameter Key Rate", fontsize=14)
    plt.xlabel("Fiber Length (km)", fontsize=12)
    plt.ylabel("Secret Key Rate per Pulse", fontsize=12)
    plt.ylim(1e-9, 1.0)
    
    from matplotlib.ticker import LogLocator
    plt.gca().yaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
    plt.grid(True, which='major', linestyle='-', alpha=0.6)
    plt.grid(True, which='minor', linestyle=':', alpha=0.3)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    
    # Create output directory
    output_dir = os.path.join(script_dir, "notebook_comparison_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save output plot
    output_plot = os.path.join(output_dir, "notebook_vs_fixed_comparison.png")
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {output_plot}")
    
    # Also copy the source JSON there for record keeping
    import shutil
    shutil.copy2(target_file, os.path.join(output_dir, "source_notebook_data.json"))
    print(f"Source data copied to {output_dir}")

if __name__ == "__main__":
    main()
