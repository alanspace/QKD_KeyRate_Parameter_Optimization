import os
import sys
import time
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import glob
from scipy.optimize import minimize

# --- PATH SETUP ---
# Determine absolute paths based on the script location
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path) # Optimization/run_scripts/JAX_variants
optimization_dir = os.path.abspath(os.path.join(script_dir, "../../")) # Optimization
project_root = os.path.dirname(optimization_dir) # Project Root
tools_dir = os.path.join(optimization_dir, "tools")

# Add paths to sys.path
if project_root not in sys.path:
    sys.path.append(project_root)
if tools_dir not in sys.path:
    sys.path.append(tools_dir)

# Import Physics & Tools
from src.qkd.model import objective_val_and_grad
import generate_timing_report

# --- CONFIGURATION ---
# JAX configuration
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu") # Force CPU for 64-bit precision

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def optimize_single_nx_sequence(n_X, L_values, bounds, initial_guess, constants):
    """
    Optimizes key rate for a sequence of fiber lengths for a fixed n_X.
    """
    alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event = constants
    
    results = []
    current_best_guess = initial_guess.copy()
    
    # Wrapper for minimize using the imported objective_val_and_grad
    def objective_wrapper(params, L_val):
        val, grad = objective_val_and_grad(
            params, L_val, n_X, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event
        )
        # Handle numerical instability
        if not np.isfinite(val): return 1e9, np.zeros_like(params)
        grad = np.clip(np.array(grad), -1e6, 1e6) # Clip gradients
        return float(val), np.array(grad)

    start_time = time.time()
    
    # Optimization Loop over Lengths
    # Using simple loop instead of tqdm parallel inside to avoid nesting complexity with process pools
    # Iterate L from 0 to 200
    for i, L in enumerate(L_values):
        try:
            candidates = []
            candidates.append(current_best_guess) # Warm start
            candidates.append(initial_guess)      # Reset
            # Jitter
            candidates.append(current_best_guess * np.random.uniform(0.99, 1.01, size=len(initial_guess)))
            
            # Deep search for danger zone
            if L > 100:
                # Add 20 diverse candidates to explore every nook and cranny
                for _ in range(20):
                    # Mix of small, medium, and large perturbations
                    scale = np.random.choice([0.05, 0.1, 0.2, 0.3]) # 5% to 30% jitter
                    perturbation = np.random.uniform(1.0 - scale, 1.0 + scale, size=len(initial_guess))
                    candidates.append(current_best_guess * perturbation)
                
                # Also try random jumps from the global default
                candidates.append(initial_guess * np.random.uniform(0.8, 1.2, size=len(initial_guess)))
            
            best_rate = -1.0
            best_params = current_best_guess
            
            for start_point in candidates:
                start_point = np.clip(start_point, [b[0] for b in bounds], [b[1] for b in bounds])
                res = minimize(
                    fun=lambda p: objective_wrapper(p, L),
                    x0=start_point,
                    method='L-BFGS-B',
                    jac=True,
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12}
                )
                rate = -res.fun
                if rate > best_rate:
                    best_rate = rate
                    best_params = res.x
            
            if best_rate > 0:
                current_best_guess = best_params.copy()
            else:
                best_rate = 0.0
                
            results.append((L, n_X, best_rate, best_params, initial_guess))
            
        except Exception as e:
            print(f"❌ Error at L={L:.1f}, n_X={n_X:.0e}: {e}")
            results.append((L, n_X, 0.0, [0.0]*5, initial_guess))

    duration = time.time() - start_time
    return results, duration

def save_detailed_plot(data, target_nx, output_dir):
    """
    Generates and saves a detailed plot for a specific n_X.
    """
    target_nx_str = str(float(target_nx))
    if target_nx_str not in data:
        return

    filtered_data = data[target_nx_str]
    # Filter non-zero
    filtered_data = [e for e in filtered_data if e["key_rate"] > 1e-20]
    
    if not filtered_data:
        return

    fiber_lengths = [e["fiber_length"] for e in filtered_data]
    key_rates = [e["key_rate"] for e in filtered_data]
    
    plt.figure(figsize=(12, 6))
    
    # Subplot 1: Key Rate
    plt.subplot(1, 2, 1)
    plt.plot(fiber_lengths, np.log10(key_rates), 'b-', label="Key Rate")
    plt.xlabel("Fiber Length (km)")
    plt.ylabel("log10(Key Rate)")
    plt.title(f"Key Rate (n_X={float(target_nx):.0e})")
    plt.ylim(-10, 0)
    plt.grid(True)
    plt.legend()
    
    # Subplot 2: Parameters
    plt.subplot(1, 2, 2)
    params = filtered_data[0]["optimized_params"].keys()
    for p in params:
        vals = [e["optimized_params"][p] for e in filtered_data]
        plt.plot(fiber_lengths, vals, label=p)
    
    plt.xlabel("Fiber Length (km)")
    plt.ylabel("Value")
    plt.title(f"Parameters (n_X={float(target_nx):.0e})")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    filename = f"qkd_detailed_nx_{float(target_nx):.0e}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

def main():
    # --- SETUP RUN DIRECTORY ---
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(script_dir, "../outputs/JAX/Baseline", f"run_BASELINE_{timestamp}")
    detailed_plots_dir = os.path.join(run_dir, "detailed_plots")
    os.makedirs(detailed_plots_dir, exist_ok=True)
    
    # --- SAVE METADATA ---
    metadata = {
        "script_name": os.path.basename(__file__),
        "strategy": "Baseline (Standard L-BFGS-B)",
        "timestamp": timestamp,
        "description": "Standard optimization using L-BFGS-B for all distances. Fast but may be susceptible to local minima at >100km."
    }
    with open(os.path.join(run_dir, "run_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"🚀 Starting Optimization Run: {timestamp}")
    print(f"📂 Output Directory: {run_dir}")
    print(f"ℹ️  Metadata saved.")

    # --- SIMULATION PARAMETERS ---
    # Fiber Lengths: 0 to 200 km
    L_values = np.linspace(0, 200, 1000)
    
    # n_X Values to sweep
    n_X_values = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
    
    # Physics Constants
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

    # Optimization Constraints
    bounds = [
        (4e-4, 0.9), (2e-4, 0.5), (1e-12, 0.999), 
        (1e-12, 0.999), (1e-12, 0.999)
    ]
    
    # Initial Guesses (Architect's Table)
    initial_guesses_map = {
        1e4: np.array([0.65, 0.15, 0.05, 0.61, 0.425]),
        1e5: np.array([0.62, 0.24, 0.10, 0.70, 0.55]),
        1e6: np.array([0.68, 0.30, 0.14, 0.74, 0.66]),
        1e7: np.array([0.55, 0.34, 0.15, 0.75, 0.75]),
        1e8: np.array([0.54, 0.375, 0.16, 0.775, 0.83]),
        1e9: np.array([0.52, 0.40, 0.18, 0.785, 0.88]),
    }
    default_guess = np.array([0.52, 0.40, 0.18, 0.785, 0.88])

    # --- EXECUTION ---
    print(f"🏎️  Running optimization for {len(n_X_values)} n_X curves over {len(L_values)} points each...")
    
    wall_start = time.time()
    total_sequential_time = 0.0
    timing_data = {}
    final_results = []
    
    # Parallel Execution
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_nx = {}
        for n_X in n_X_values:
            guess = initial_guesses_map.get(n_X, default_guess)
            future = executor.submit(optimize_single_nx_sequence, n_X, L_values, bounds, guess, constants)
            future_to_nx[future] = n_X
            
        for future in concurrent.futures.as_completed(future_to_nx):
            n_X = future_to_nx[future]
            try:
                # Unpack result
                results, duration = future.result()
                
                # Store Data
                final_results.extend(results)
                timing_data[str(float(n_X))] = duration
                total_sequential_time += duration
                
                print(f"✅ Finished n_X={n_X:.0e} in {duration/60:.2f} min")
                
            except Exception as e:
                print(f"❌ Exception for n_X={n_X}: {e}")

    wall_end = time.time()
    wall_duration = wall_end - wall_start
    print(f"\n🏁 Optimization Complete in {wall_duration/60:.2f} min (Wall Time)")

    # --- DATA PROCESSING ---
    # 1. Raw Results List
    raw_dataset = [{
        "fiber_length": float(r[0]),
        "n_X": int(r[1]),
        "key_rate": float(r[2]),
        "optimized_parameters": {
            "mu_1": float(r[3][0]), "mu_2": float(r[3][1]), "P_mu_1": float(r[3][2]), 
            "P_mu_2": float(r[3][3]), "P_X_value": float(r[3][4])
        },
        "initial_guess": {
             "mu_1": float(r[4][0]), "mu_2": float(r[4][1]), "P_mu_1": float(r[4][2]), 
             "P_mu_2": float(r[4][3]), "P_X_value": float(r[4][4])
        }
    } for r in final_results]
    
    # Save Raw
    raw_path = os.path.join(run_dir, f"qkd_optimization_results_{timestamp}.json")
    with open(raw_path, 'w') as f:
        json.dump(raw_dataset, f, indent=2)
    print(f"💾 Raw results saved to: {raw_path}")

    # 2. Grouped Dataset (Reordered)
    grouped_data = {}
    for entry in raw_dataset:
        n_X = float(entry["n_X"])
        # Calculate derived physics parameters for reference (e1..e4)
        # e_1 = L/100, e_2 = -log(P_dc), e_3 = e_mis*100, e_4 = log(n_X)
        e_1 = entry["fiber_length"] / 100
        e_2 = -np.log10(P_dc_value)
        e_3 = e_mis * 100
        e_4 = np.log10(n_X)
        
        enhanced_entry = {
            "fiber_length": entry["fiber_length"],
            "e_1": e_1, "e_2": e_2, "e_3": e_3, "e_4": e_4,
            "key_rate": entry["key_rate"],
            "optimized_params": entry["optimized_parameters"]
        }
        
        nx_str = str(n_X)
        if nx_str not in grouped_data: grouped_data[nx_str] = []
        grouped_data[nx_str].append(enhanced_entry)
    
    # Sort by length
    for nx in grouped_data:
        grouped_data[nx].sort(key=lambda x: x["fiber_length"])
        
    # Save Grouped
    grouped_path = os.path.join(run_dir, f"qkd_grouped_dataset_{timestamp}.json")
    with open(grouped_path, 'w') as f:
        json.dump(grouped_data, f, indent=2)
    print(f"💾 Grouped dataset saved to: {grouped_path}")

    # --- REPORTING ---
    print("\n📊 Generating Reports...")
    
    # 1. Timing Report
    try:
        generate_timing_report.generate_report(
            timing_data=timing_data,
            total_sequential=total_sequential_time,
            wall_time=wall_duration,
            num_cores=os.cpu_count(),
            output_dir=run_dir
        )
    except Exception as e:
        print(f"⚠️ Failed to generate timing report: {e}")

    # 2. Detailed Plots
    print("   Generating detailed plots...")
    for nx in grouped_data.keys():
        try:
            save_detailed_plot(grouped_data, nx, detailed_plots_dir)
        except Exception as e:
            print(f"   ⚠️ Plot error for n_X={nx}: {e}")
            
    print(f"\n✅ Run {timestamp} Finished Successfully.")
    print(f"👉 Results in: {run_dir}")

if __name__ == "__main__":
    main()
