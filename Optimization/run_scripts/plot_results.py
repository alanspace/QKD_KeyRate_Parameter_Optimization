import os
import sys
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

# Setup Path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

# Try importing the physics model
try:
    from src.qkd.model import scalar_objective
except ImportError as e:
    print(f"Error importing physics model: {e}")
    sys.exit(1)

# Constants (Must match JAX script exactly)
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

# Fixed Parameters Map (Architect's Table)
INITIAL_GUESSES_MAP = {
    1e4: np.array([0.65, 0.15, 0.05, 0.61, 0.425]),
    1e5: np.array([0.62, 0.24, 0.10, 0.70, 0.55]),
    1e6: np.array([0.68, 0.30, 0.14, 0.74, 0.66]),
    1e7: np.array([0.55, 0.34, 0.15, 0.75, 0.75]),
    1e8: np.array([0.54, 0.375, 0.16, 0.775, 0.83]),
    1e9: np.array([0.52, 0.40, 0.18, 0.785, 0.88]),
}
DEFAULT_GUESS = np.array([0.52, 0.40, 0.18, 0.785, 0.88])

def find_latest_json(directory):
    # Find files matching pattern
    pattern = os.path.join(directory, "reordered_qkd_results_jax_*.json")
    files = glob.glob(pattern)
    if not files: return None
    # Sort by modification time
    return max(files, key=os.path.getmtime)

def calculate_fixed_curve(L_values, n_X, fixed_params):
    rates = []
    # Make sure fixed_params is JAX array
    p_jax = jnp.array(fixed_params)
    for L in L_values:
        # scalar_objective call
        try:
            r = scalar_objective(p_jax, float(L), float(n_X), *constants)
            # scalar_objective returns negative rate (minimization target)
            # We must flip it back to positive for plotting
            rates.append(-float(r))
        except Exception as e:
            print(f"Error calcing fixed rate at L={L}: {e}")
            rates.append(0.0)
    return np.array(rates)

def generate_plots(json_path, filename_prefix=""):
    if not os.path.exists(json_path):
        print(f"Error: File {json_path} not found.")
        return

    results_dir = os.path.dirname(json_path)
    print(f"Loading results from {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Colors (Tab10 / Notebook Style)
    # Use high-contrast manual colors (Red, Green, Blue, Purple, Cyan, Orange)
    manual_colors = ['red', 'green', 'blue', 'purple', 'cyan', 'orange']
    # Ensure we have enough colors if more keys exist (cycle them)
    import itertools
    # Generate list of n_X values present
    nx_keys = sorted(data.keys(), key=lambda x: float(x))
    line_colors = list(itertools.islice(itertools.cycle(manual_colors), len(nx_keys)))
    
    # 1. Summary Plot (Optimized vs Fixed) for ALL n_X
    # ------------------------------------------------
    print("Generating Summary Plot (Optimized vs Fixed)...")
    plt.figure(figsize=(10, 8))
    
    for i, nx_str in enumerate(nx_keys):
        n_X = float(nx_str)
        if n_X not in INITIAL_GUESSES_MAP: continue 
        
        entries = data[nx_str]
        entries.sort(key=lambda x: x["fiber_length"])
        
        L = np.array([e["fiber_length"] for e in entries])
        R_opt = np.array([e["key_rate"] for e in entries])
        
        fixed_params = INITIAL_GUESSES_MAP.get(n_X, DEFAULT_GUESS)
        R_fixed = calculate_fixed_curve(L, n_X, fixed_params)
        
        color = line_colors[i]
        plt.semilogy(L, R_opt, '-', color=color, linewidth=2, label=f'Optimized $n_X=10^{int(np.log10(n_X))}$')
        plt.semilogy(L, R_fixed, '--', color=color, linewidth=1.5, alpha=0.7) 
        plt.semilogy([], [], '--', color=color, label=f'Fixed $n_X=10^{int(np.log10(n_X))}$') 

    plt.title("Optimized vs. Fixed-Parameter Key Rate for Different $n_X$ Values", fontsize=14)
    plt.xlabel("Fiber Length (km)", fontsize=12)
    plt.ylabel("Secret Key Rate per Pulse", fontsize=12)
    plt.ylim(1e-9, 1.0) 
    
    from matplotlib.ticker import LogLocator
    plt.gca().yaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
    plt.grid(True, which='major', linestyle='-', alpha=0.6)
    plt.grid(True, which='minor', linestyle=':', alpha=0.3)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    output_summary = os.path.join(results_dir, f"{filename_prefix}summary_optimized_vs_fixed.png")
    plt.savefig(output_summary, dpi=300, bbox_inches='tight')
    print(f"Saved summary plot to {output_summary}")
    plt.close()
    
    # 2. Per-n_X Detailed Plots (Rate + Parameters)
    # ------------------------------------------------
    output_dir_detailed = os.path.join(results_dir, "detailed_plots")
    os.makedirs(output_dir_detailed, exist_ok=True)
    
    print("Generating Detailed Per-n_X Plots...")
    for nx_str in nx_keys:
        n_X = float(nx_str)
        entries = data[nx_str]
        entries.sort(key=lambda x: x["fiber_length"])
        
        L = np.array([e["fiber_length"] for e in entries])
        
        mu_1 = np.array([e["optimized_parameters"]["mu_1"] for e in entries])
        mu_2 = np.array([e["optimized_parameters"]["mu_2"] for e in entries])
        P_mu_1 = np.array([e["optimized_parameters"]["P_mu_1"] for e in entries])
        P_mu_2 = np.array([e["optimized_parameters"]["P_mu_2"] for e in entries])
        P_X = np.array([e["optimized_parameters"]["P_X_value"] for e in entries])
        
        R_opt = np.array([e["key_rate"] for e in entries])
        
        fig, axs = plt.subplots(1, 2, figsize=(18, 6))
        
        axs[0].semilogy(L, R_opt, 'k-', linewidth=2, label='Optimized Rate')
        fixed_params = INITIAL_GUESSES_MAP.get(n_X, DEFAULT_GUESS)
        R_fixed = calculate_fixed_curve(L, n_X, fixed_params)
        axs[0].semilogy(L, R_fixed, 'r--', linewidth=1.5, label='Fixed Parameters')
        
        axs[0].set_ylim(1e-10, 1.0)
        axs[0].set_title(f"Key Rate ($n_X=10^{int(np.log10(n_X))}$)", fontsize=14)
        axs[0].set_xlabel("Fiber Length (km)")
        axs[0].set_ylabel("Key Rate")
        axs[0].grid(True, which="both", alpha=0.5)
        axs[0].legend()
        
        # Parameters (Width Hierarchy for Overlap Visibility: Thick Intensities, Thin Probabilities)
        axs[1].plot(L, mu_1, color='r', linewidth=4, label=r'$\mu_1$')
        axs[1].plot(L, mu_2, color='g', linewidth=4, label=r'$\mu_2$')
        axs[1].plot(L, P_mu_1, color='b', linewidth=2, label=r'$P_{\mu_1}$')
        axs[1].plot(L, P_mu_2, color='c', linewidth=2, label=r'$P_{\mu_2}$')
        axs[1].plot(L, P_X, color='k', linewidth=3, label=r'$P_X$')
        
        axs[1].set_title(f"Optimized Parameters (Dynamic)", fontsize=14)
        axs[1].set_xlabel("Fiber Length (km)")
        axs[1].set_ylabel("Parameter Value")
        axs[1].set_ylim(0, 1.0)
        axs[1].grid(True, alpha=0.5)
        axs[1].legend(loc='upper right')
        
        filename = f"{filename_prefix}detailed_nx_{nx_str}.png"
        path = os.path.join(output_dir_detailed, filename)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
    print(f"Detailed plots saved to {output_dir_detailed}")

def main():
    results_dir = os.path.join(script_dir, "results_global_jax")
    json_path = find_latest_json(results_dir)
    if not json_path:
        print("No results found in results_global_jax.")
        return
    generate_plots(json_path)

if __name__ == "__main__":
    main()
