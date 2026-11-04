import os
import sys
import time
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import glob
from scipy.optimize import minimize, dual_annealing

# --- PATH SETUP ---
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path) # Optimization/run_scripts/JAX_variants
optimization_dir = os.path.abspath(os.path.join(script_dir, "../../")) # Optimization
project_root = os.path.dirname(optimization_dir) # Project Root
tools_dir = os.path.join(optimization_dir, "tools")

if project_root not in sys.path: sys.path.append(project_root)
if tools_dir not in sys.path: sys.path.append(tools_dir)

from src.qkd.model import objective_val_and_grad, scalar_objective
import generate_timing_report

# --- CONFIGURATION ---
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Compiled objective for Dual Annealing (value only, no gradients)
objective_val_only = jax.jit(scalar_objective)

def save_partial_result(output_dir, n_X, results):
    """Saves the results for a single n_X to a JSON file."""
    filename = os.path.join(output_dir, f"partial_result_nX_{float(n_X):.0e}.json")
    
    # Convert to JSON-serializable format
    formatted_data = []
    for r in results:
        # r structure: (L, n_X, best_rate, best_params, initial_guess)
        entry = {
            "fiber_length": float(r[0]),
            "n_X": float(r[1]),
            "key_rate": float(r[2]),
            "optimized_parameters": {
                "mu_1": float(r[3][0]), "mu_2": float(r[3][1]), "P_mu_1": float(r[3][2]), 
                "P_mu_2": float(r[3][3]), "P_X_value": float(r[3][4])
            }
        }
        formatted_data.append(entry)
        
    with open(filename, 'w') as f:
        json.dump(formatted_data, f, indent=2)
    print(f"💾 Saved partial result to {filename}")

def optimize_single_nx_sequence_pure_da(n_X, L_values, bounds, initial_guess, constants, output_dir):
    """
    Pure Dual Annealing Strategy:
    - L <= 100 km: Fast L-BFGS-B (Standard)
    - L > 100 km:  PURE Dual Annealing (No Local Search, High Iterations)
    """
    alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event = constants
    
    results = []
    current_best_guess = initial_guess.copy()
    
    # Wrapper for minimize (L-BFGS-B)
    def objective_wrapper_local(params, L_val):
        val, grad = objective_val_and_grad(
            params, L_val, n_X, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event
        )
        if not np.isfinite(val): return 1e9, np.zeros_like(params)
        grad = np.clip(np.array(grad), -1e6, 1e6)
        return float(val), np.array(grad)

    # Wrapper for dual_annealing (Value only)
    def objective_wrapper_global(params, L_val):
        val = objective_val_only(
            jnp.array(params), L_val, n_X, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event
        )
        if not np.isfinite(val): return 1e9
        return float(val)

    start_time = time.time()
    
    for i, L in enumerate(L_values):
        try:
            best_rate = -1.0
            best_params = current_best_guess
            
            # --- STRATEGY SWITCH ---
            # We use Standard (Fast) for the easy part (<= 100km) to save time
            if L <= 100:
                res = minimize(
                    fun=lambda p: objective_wrapper_local(p, L),
                    x0=current_best_guess,
                    method='L-BFGS-B',
                    jac=True,
                    bounds=bounds,
                    options={'maxiter': 500, 'ftol': 1e-12}
                )
                rate = -res.fun
                if rate > best_rate:
                    best_rate = rate
                    best_params = res.x

            else:
                # === PURE DUAL ANNEALING (L > 100km) ===
                # "Stepped Iteration Strategy" to reduce jitter at long distances
                if L <= 120:
                    iter_count = 5000
                elif L <= 140:
                    iter_count = 7500
                elif L <= 160:
                    iter_count = 10000
                else:
                    iter_count = 15000 # Massive effort for >160km

                res = dual_annealing(
                    func=lambda p: objective_wrapper_global(p, L),
                    bounds=bounds,
                    maxiter=iter_count,
                    initial_temp=5230.0,
                    visit=2.62,
                    accept=-5.0,
                    x0=current_best_guess,
                    no_local_search=True, # STRICTLY PURE
                )
                rate = -res.fun
                # Note: DA result might exceed bounds slightly or be slightly off-peak
                if rate > best_rate:
                    best_rate = rate
                    best_params = res.x
            
            if L > 100 and i % 5 == 0:
                print(f"   [n_X={n_X:.0e}] L={L:.1f}km -> Rate={best_rate:.2e} (Pure DA, maxiter={iter_count})")

        except Exception as e:
            print(f"❌ Error at L={L:.1f}, n_X={n_X:.0e}: {e}")
            results.append((L, n_X, 0.0, [0.0]*5, initial_guess))

    duration = time.time() - start_time
    
    # --- ROBUST SAVING ---
    try:
        save_partial_result(output_dir, n_X, results)
    except Exception as e:
        print(f"❌ Failed to save partial result for n_X={n_X:.0e}: {e}")

    return results, duration

def save_detailed_plot(data, target_nx, output_dir):
    target_nx_str = str(float(target_nx))
    if target_nx_str not in data: return
    filtered_data = [e for e in data[target_nx_str] if e["key_rate"] > 1e-20]
    if not filtered_data: return

    fiber_lengths = [e["fiber_length"] for e in filtered_data]
    key_rates = [e["key_rate"] for e in filtered_data]
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(fiber_lengths, np.log10(key_rates), 'b-', label="Key Rate")
    plt.xlabel("Fiber Length (km)")
    plt.ylabel("log10(Key Rate)")
    plt.title(f"Key Rate (n_X={float(target_nx):.0e})")
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    params = filtered_data[0]["optimized_params"].keys()
    for p in params:
        vals = [e["optimized_params"][p] for e in filtered_data]
        plt.plot(fiber_lengths, vals, label=p)
    plt.xlabel("Fiber Length (km)")
    plt.ylabel("Value")
    plt.title(f"Parameters")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"qkd_detailed_nx_{float(target_nx):.0e}.png"), dpi=300)
    plt.close()

def main():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(script_dir, "../outputs/JAX/Pure_DA", f"run_PURE_DA_{timestamp}")
    detailed_plots_dir = os.path.join(run_dir, "detailed_plots")
    os.makedirs(detailed_plots_dir, exist_ok=True)
    
    # --- SAVE METADATA ---
    metadata = {
        "script_name": os.path.basename(__file__),
        "strategy": "Pure Dual Annealing",
        "timestamp": timestamp,
        "description": "Standard optimization for L<=100km. PURE Dual Annealing (Global Search, No Local Search) for L>100km."
    }
    with open(os.path.join(run_dir, "run_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"🚀 Starting PURE DUAL ANNEALING Run: {timestamp}")
    print(f"📂 Output Directory: {run_dir}")
    print("ℹ️  Strategy: L>100km uses Pure Dual Annealing (no_local_search=True, maxiter=5000)")

    L_values = np.linspace(0, 200, 400) # 0.5km steps
    n_X_values = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
    # n_X_values = [1e9] # Test Only
    
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

    bounds = [(4e-4, 0.9), (2e-4, 0.5), (1e-12, 0.999), (1e-12, 0.999), (1e-12, 0.999)]
    
    # Initial Guesses
    initial_guesses_map = {
        1e4: np.array([0.65, 0.15, 0.05, 0.61, 0.425]),
        1e9: np.array([0.52, 0.40, 0.18, 0.785, 0.88]),
    }
    default_guess = np.array([0.52, 0.40, 0.18, 0.785, 0.88])

    print(f"🏎️  Running optimization...")
    wall_start = time.time()
    total_sequential_time = 0.0
    timing_data = {}
    final_results = []
    
    # Using ProcessPoolExecutor to truly parallelize (Pure DA is CPU heavy)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_nx = {}
        for n_X in n_X_values:
            guess = initial_guesses_map.get(n_X, default_guess)
            future = executor.submit(optimize_single_nx_sequence_pure_da, n_X, L_values, bounds, guess, constants, run_dir)
            future_to_nx[future] = n_X
            
        for future in concurrent.futures.as_completed(future_to_nx):
            n_X = future_to_nx[future]
            try:
                results, duration = future.result()
                final_results.extend(results)
                timing_data[str(float(n_X))] = duration
                total_sequential_time += duration
                print(f"✅ Finished n_X={n_X:.0e} in {duration/60:.2f} min")
            except Exception as e:
                print(f"❌ Exception for n_X={n_X}: {e}")

    wall_end = time.time()
    wall_duration = wall_end - wall_start
    print(f"\n🏁 Complete in {wall_duration/60:.2f} min (Wall Time)")

    # Save Results
    # --- RECOVERY LOGIC ---
    if not final_results:
        print("\n⚠️  WARNING: Main process received empty results (Serialization Issue?).")
        print("🔎  Attempting to recover from partial JSON files...")
        partial_files = glob.glob(os.path.join(run_dir, "partial_result_nX_*.json"))
        if partial_files:
            print(f"✅ Found {len(partial_files)} partial files. Merging...")
            # Reconstruct 'raw_dataset'-like structure directly from JSONs
            # Note: The partial JSONs already have the 'formatted' structure (dict), 
            # whereas 'final_results' was list-of-tuples.
            # We can skip the 'raw_dataset' list comp loop below if we load from files.
            final_results_from_files = []
            for pf in partial_files:
                try:
                    with open(pf, 'r') as f:
                        data = json.load(f)
                        final_results_from_files.extend(data)
                except Exception as e:
                    print(f"❌ Error reading {pf}: {e}")
            
            # If we recovered data, we need to adapt the downstream logic.
            # The downstream logic expects 'raw_dataset' to be created from 'final_results' tuples.
            # OR 'final_results' to be populated.
            # Let's populate 'raw_dataset' directly since we have the data in that format.
            raw_dataset = final_results_from_files
        else:
            print("❌ No partial files found. Recovery failed.")
            raw_dataset = [] # Empty
    else:
        # Standard Path: Convert tuples to dicts
        raw_dataset = [{
            "fiber_length": float(r[0]),
            "n_X": int(r[1]),
            "key_rate": float(r[2]),
            "optimized_parameters": {
                "mu_1": float(r[3][0]), "mu_2": float(r[3][1]), "P_mu_1": float(r[3][2]), 
                "P_mu_2": float(r[3][3]), "P_X_value": float(r[3][4])
            }
        } for r in final_results]
    
    grouped_data = {}
    for entry in raw_dataset:
        n_X = float(entry["n_X"])
        entry["e_1"] = entry["fiber_length"] / 100
        entry["e_2"] = -np.log10(P_dc_value)
        entry["e_3"] = e_mis * 100
        entry["e_4"] = np.log10(n_X)
        # Entry already has 'optimized_parameters' key from save_partial_result OR list comp
        entry["optimized_params"] = entry["optimized_parameters"]
        nx_str = str(n_X)
        if nx_str not in grouped_data: grouped_data[nx_str] = []
        grouped_data[nx_str].append(entry)
    
    for nx in grouped_data: grouped_data[nx].sort(key=lambda x: x["fiber_length"])
        
    grouped_path = os.path.join(run_dir, f"qkd_grouped_dataset_{timestamp}.json")
    with open(grouped_path, 'w') as f: json.dump(grouped_data, f, indent=2)

    try: generate_timing_report.generate_report(timing_data, total_sequential_time, wall_duration, os.cpu_count(), run_dir)
    except: pass

    for nx in grouped_data.keys():
        try: save_detailed_plot(grouped_data, nx, detailed_plots_dir)
        except: pass

if __name__ == "__main__":
    main()
