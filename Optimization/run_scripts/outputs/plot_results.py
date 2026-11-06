import os
import sys
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import config

# Force CPU for plotting to avoid memory conflicts with the main optimization process
config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

# Setup Path to find src.qkd
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

# Try importing the physics model
# We need this to calculate the "Fixed" curve for comparison
try:
    from src.qkd.model import scalar_objective
except ImportError as e:
    print(f"⚠️ Error importing physics model for plotting: {e}")
    print("Plotting will continue, but 'Fixed Parameter' curves might fail.")

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

# Fixed Parameters Map (Architect's Table - Baseline)
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
    """
    Calculates the key rate vs distance for a FIXED set of parameters.
    Used to generate the dashed comparison line.
    """
    rates = []
    # Ensure params are JAX-compatible
    p_jax = jnp.array(fixed_params)
    
    for L in L_values:
        try:
            # scalar_objective returns NEGATIVE key rate (minimization target)
            r = scalar_objective(p_jax, float(L), float(n_X), *constants)
            
            # We assume rate is 0 if r is NaN or positive (meaning negative key rate)
            val = -float(r)
            if np.isnan(val) or val < 0:
                val = 0.0
            rates.append(val)
        except Exception:
            rates.append(0.0)
            
    return np.array(rates)

def generate_plots(json_path, filename_prefix=""):
    """
    Main plotting function.
    Generates:
    1. Summary Plot (All n_X curves, Optimized vs Fixed)
    2. Detailed Plot per n_X (Key Rate + Parameter Evolution)
    """
    if not os.path.exists(json_path):
        print(f"Error: File {json_path} not found.")
        return

    results_dir = os.path.dirname(json_path)
    print(f"📊 Loading results from {json_path}")
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to load JSON: {e}")
        return

    # Colors for different n_X values
    manual_colors = ['red', 'green', 'blue', 'purple', 'cyan', 'orange', 'brown', 'pink']
    
    # Sort n_X keys numerically
    try:
        nx_keys = sorted(data.keys(), key=lambda x: float(x))
    except:
        nx_keys = list(data.keys())

    import itertools
    line_colors = list(itertools.islice(itertools.cycle(manual_colors), len(nx_keys)))
    
    # --- 1. Summary Plot (Optimized vs Fixed) ---
    print("   ... Generating Summary Plot")
    plt.figure(figsize=(12, 9))
    
    for i, nx_str in enumerate(nx_keys):
        n_X = float(nx_str)
        
        entries = data[nx_str]
        # Sort entries by fiber length
        entries.sort(key=lambda x: x["fiber_length"])
        
        L = np.array([e["fiber_length"] for e in entries])
        R_opt = np.array([e["key_rate"] for e in entries])
        
        # Calculate Fixed Baseline
        fixed_params = INITIAL_GUESSES_MAP.get(n_X, DEFAULT_GUESS)
        R_fixed = calculate_fixed_curve(L, n_X, fixed_params)
        
        # Determine label exponent
        try:
            exp_label = int(np.log10(n_X))
        except:
            exp_label = str(n_X)

        color = line_colors[i]
        
        # Plot Optimized
        plt.semilogy(L, R_opt, '-', color=color, linewidth=2.5, 
                     label=f'Opt $10^{{{exp_label}}}$')
        
        # Plot Fixed (Dashed)
        plt.semilogy(L, R_fixed, '--', color=color, linewidth=1.5, alpha=0.6,
                     label=f'Fix $10^{{{exp_label}}}$') 

    plt.title(f"Optimized vs. Fixed-Parameter Key Rate ({filename_prefix.strip('_')})", fontsize=16)
    plt.xlabel("Fiber Length (km)", fontsize=14)
    plt.ylabel("Secret Key Rate (bits/pulse)", fontsize=14)
    plt.ylim(1e-10, 1.0)
    plt.xlim(0, 200)
    
    plt.grid(True, which='major', linestyle='-', alpha=0.6)
    plt.grid(True, which='minor', linestyle=':', alpha=0.3)
    
    # Legend setup
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
    plt.tight_layout()
    
    output_summary = os.path.join(results_dir, f"{filename_prefix}summary_optimized_vs_fixed.png")
    plt.savefig(output_summary, dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- 2. Per-n_X Detailed Plots ---
    output_dir_detailed = os.path.join(results_dir, "detailed_plots")
    os.makedirs(output_dir_detailed, exist_ok=True)
    
    print("   ... Generating Detailed Per-n_X Plots")
    for nx_str in nx_keys:
        n_X = float(nx_str)
        entries = data[nx_str]
        entries.sort(key=lambda x: x["fiber_length"])
        
        L = np.array([e["fiber_length"] for e in entries])
        
        # Extract Parameters
        mu_1 = np.array([e["optimized_parameters"]["mu_1"] for e in entries])
        mu_2 = np.array([e["optimized_parameters"]["mu_2"] for e in entries])
        P_mu_1 = np.array([e["optimized_parameters"]["P_mu_1"] for e in entries])
        P_mu_2 = np.array([e["optimized_parameters"]["P_mu_2"] for e in entries])
        P_X = np.array([e["optimized_parameters"]["P_X_value"] for e in entries])
        
        R_opt = np.array([e["key_rate"] for e in entries])
        
        # Recalculate fixed for this specific plot
        fixed_params = INITIAL_GUESSES_MAP.get(n_X, DEFAULT_GUESS)
        R_fixed = calculate_fixed_curve(L, n_X, fixed_params)
        
        # Setup subplot (1x2)
        fig, axs = plt.subplots(1, 2, figsize=(18, 6))
        
        # LEFT: Key Rate
        axs[0].semilogy(L, R_opt, 'k-', linewidth=2.5, label='Optimized Rate')
        axs[0].semilogy(L, R_fixed, 'r--', linewidth=1.5, label='Fixed Params')
        
        axs[0].set_ylim(1e-10, 1.0)
        try:
            axs[0].set_title(f"Key Rate ($n_X=10^{{{int(np.log10(n_X))}}}$)", fontsize=14)
        except:
            axs[0].set_title(f"Key Rate ($n_X={n_X}$)", fontsize=14)
            
        axs[0].set_xlabel("Fiber Length (km)", fontsize=12)
        axs[0].set_ylabel("Key Rate", fontsize=12)
        axs[0].grid(True, which="both", alpha=0.5)
        axs[0].legend()
        
        # RIGHT: Parameters
        axs[1].plot(L, mu_1, color='r', linewidth=3, label=r'$\mu_1$')
        axs[1].plot(L, mu_2, color='g', linewidth=3, label=r'$\mu_2$')
        axs[1].plot(L, P_mu_1, color='b', linewidth=2, label=r'$P_{\mu_1}$', linestyle='--')
        axs[1].plot(L, P_mu_2, color='c', linewidth=2, label=r'$P_{\mu_2}$', linestyle='--')
        axs[1].plot(L, P_X, color='k', linewidth=3, label=r'$P_X$')
        
        axs[1].set_title("Optimized Parameters (Evolution)", fontsize=14)
        axs[1].set_xlabel("Fiber Length (km)", fontsize=12)
        axs[1].set_ylabel("Parameter Value", fontsize=12)
        axs[1].set_ylim(0, 1.0)
        axs[1].grid(True, alpha=0.5)
        axs[1].legend(loc='upper right')
        
        filename = f"{filename_prefix}detailed_nx_{nx_str}.png"
        path = os.path.join(output_dir_detailed, filename)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        
    print(f"✅ All plots saved to {results_dir}")

if __name__ == "__main__":
    # Test mode: try to find latest file in results_global_jax and plot
    # This block allows running this script standalone
    script_dir_loc = os.path.dirname(os.path.abspath(__file__))
    # tier3 outputs to results_global_jax in the same folder as the script
    possible_dir = os.path.join(script_dir_loc, "results_global_jax")
    
    # Try finding files in subdirectories (run_*)
    # Pattern: results_global_jax/run_*/reordered_qkd_results_jax_*.json
    pattern = os.path.join(possible_dir, "run_*", "reordered_qkd_results_jax_*.json")
    files = glob.glob(pattern)
    
    if files:
        latest = max(files, key=os.path.getmtime)
        print(f"Test run found: {latest}")
        generate_plots(latest, filename_prefix="TEST_")
    else:
        print("No results file found for standalone test.")
