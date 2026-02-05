import os
import sys
import time
import json
import functools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
import concurrent.futures
from collections import defaultdict
from scipy.optimize import minimize, dual_annealing, differential_evolution, Bounds
import glob
import logging

# Ensure the script uses the correct project root
script_dir = os.path.dirname(os.path.abspath(__file__))
# Go up two levels: Optimization/script_version -> Optimization -> Project Root
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

# Add tools to path
tools_dir = os.path.join(os.path.dirname(script_dir), 'tools')
if tools_dir not in sys.path:
    sys.path.append(tools_dir)
import generate_timing_report

# Import project modules
from src.qkd.model import calculate_key_rates_and_metrics, objective, penalty, objective_val_and_grad
from src.qkd.physics import (
    calculate_eta_ch, calculate_eta_sys, calculate_D_mu_k, 
    calculate_n_X_total, calculate_N, calculate_n_Z_total,
    calculate_e_mu_k, calculate_e_obs, calculate_h, calculate_lambda_EC,
    calculate_sqrt_term, calculate_n_pm, calculate_S_0, calculate_S_1,
    calculate_m_mu_k, calculate_m_pm, calculate_v_1, calculate_gamma,
    calculate_Phi, calculate_l, calculate_R
)

# JAX imports
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.scipy.special import logsumexp, gamma
from jax.experimental import pjit
from jax.sharding import Mesh

# FORCE CPU for 64-bit precision (Metal/GPU lacks f64 support currently)
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def optimal_parameters(params):
    mu_1, mu_2, P_mu_1, P_mu_2, P_X_value = params
    mu_3 = 2e-4
    P_mu_3 = 1 - P_mu_1 - P_mu_2
    P_Z_value = 1 - P_X_value
    mu_k_values = jnp.array([mu_1, mu_2, mu_3])
    return params, mu_3, P_mu_3, P_Z_value, mu_k_values

def optimize_single_nx_sequence(n_X, L_values, bounds, initial_guess, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event):
    start_time = time.time()
    results = []
    current_best_guess = initial_guess.copy()
    
    # Pre-compile objective
    def objective_wrapper(params, L_val):
        val, grad = objective_val_and_grad(
            params, L_val, n_X, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event
        )
        if not np.isfinite(val): return 1e9
        return float(val)

    # Gradient wrapper for L-BFGS-B (only if needed)
    def objective_grad_wrapper(params, L_val):
        val, grad = objective_val_and_grad(
             params, L_val, n_X, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event
        )
        if not np.isfinite(val): return 1e9, np.zeros_like(params)
        grad = np.clip(np.array(grad), -1e6, 1e6)
        return float(val), np.array(grad)

    # True Candidate Racing for Local Script
    # 1. Cold Start with Correct Guess
    last_success_params = initial_guesses_map.get(n_X, initial_guess).copy()
    
    # Standard Reset Guess
    default_guess = np.array([0.52, 0.40, 0.18, 0.785, 0.88])
    current_best_guess = last_success_params
    
    for L in tqdm(L_values, desc=f"Optimizing n_X={n_X:.0e}", position=0, leave=True):
        try:
            # --- STRATEGY: TRUE CANDIDATE RACING ---
            candidates = []
            
            # Candidate 1: Warm Start
            candidates.append(current_best_guess)
            
            # Candidate 2: Global Reset
            candidates.append(initial_guesses_map.get(n_X, default_guess))
            
            # Strategic Jitter Candidates
            n_candidates = 4 + (10 if L > 100 else 0)
            current_count = len(candidates)
            base_for_jitter = current_best_guess
            
            for i in range(current_count, n_candidates):
                scale = 0.1 + (0.3 if L > 150 else 0)
                perturbation = np.random.uniform(1.0 - scale, 1.0 + scale, size=len(base_for_jitter))
                candidates.append(base_for_jitter * perturbation)
                
            best_key_rate = -1.0
            best_params = None
            
            # --- RACE ---
            for start_point in candidates:
                # Bounds check
                sp = np.clip(start_point, [b[0] for b in bounds], [b[1] for b in bounds])
                try:
                    res = minimize(
                        fun=lambda p: objective_grad_wrapper(p, L), # Use grad wrapper for L-BFGS-B
                        x0=sp,
                        method='L-BFGS-B',
                        jac=True,
                        bounds=bounds,
                        options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 1500} # Strict Tolerances
                    )
                    rate = -res.fun
                    if rate > best_key_rate and res.success:
                        best_key_rate = rate
                        best_params = res.x
                except: continue
                
            # --- UPDATE HISTORY ---
            if best_key_rate < 0: best_key_rate = 0.0
            
            if best_key_rate > 1e-20:
                current_best_guess = best_params.copy()
            
            results.append((L, n_X, best_key_rate, best_params, initial_guess))

        except Exception as e:
            print(f"❌ Error at L={L}, n_X={n_X}: {e}")
            results.append((L, n_X, 0.0, [0.0]*5, initial_guess))
            
    end_time = time.time()
    duration = end_time - start_time
    return results, duration

def reorder_json_by_fiber_length(file_path):
    """
    Reorders the JSON data from a file based on the 'fiber_length'
    field within the first list found in the dictionary.
    """
    try:
        with open(file_path, 'r') as f:
            json_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at path: {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from file: {e}")
        return None

    if not isinstance(json_data, dict):
        print("Error: Input must be a dictionary.")
        return None

    # Find the first list in the dictionary
    data_list = None
    target_key = None
    for key, value in json_data.items():
        if isinstance(value, list):
            data_list = value
            target_key = key
            break  # Stop after finding the first list

    if data_list is None:
        print("Error: No list found in the dictionary.")
        return None

    # Validate the list elements
    for item in data_list:
        if not isinstance(item, dict) or "fiber_length" not in item:
            print("Error: List elements must be dictionaries containing the key 'fiber_length'.")
            return None

    # Sort the list by 'fiber_length'
    sorted_data_list = sorted(data_list, key=lambda x: x["fiber_length"])

    # Create a new dictionary with the sorted list
    reordered_json_data = json_data.copy() # Avoid modifying the original
    reordered_json_data[target_key] = sorted_data_list

    return reordered_json_data

def plot_for_nx(data, target_nx, output_dir_plots):
    """
    Plot results for a specific n_X value, filtering out zero key rates.
    """
    # Convert to string for dictionary key lookup
    target_nx_str = str(float(target_nx))  # Ensure it matches JSON key format

    # Retrieve data correctly from grouped dictionary
    if target_nx_str not in data:
        print(f"No data found for n_X = {target_nx}")
        return
    
    filtered_data = data[target_nx_str]  # Get the list of entries

    if not filtered_data:
        print(f"No data found for n_X = {target_nx}")
        return

    # Extract data
    fiber_lengths = [entry["fiber_length"] for entry in filtered_data]
    key_rates = np.array([entry["key_rate"] for entry in filtered_data])
    
    # Handle zeros for log plot (replace 0 with minimal value for plotting)
    safe_key_rates = np.where(key_rates > 0, key_rates, 1e-30)

    # ✅ Improved visualization (1x2 Layout matched to JAX/Global script)
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"LOCAL SEARCH: Results (n_X = {target_nx:.0e})", fontsize=16)
    
    # Left: Key Rate
    ax_main = axs[0]
    ax_main.plot(fiber_lengths, np.log10(safe_key_rates), linestyle='-', color='b', label="Key Rate")
    ax_main.set_xlabel("Fiber Length (km)")
    ax_main.set_ylabel("log10(Key Rate)")
    ax_main.set_title("Log10(Key Rate)")
    ax_main.set_ylim(-9.0, 0.5)
    ax_main.legend()
    ax_main.grid(True, which='both', linestyle='--')
    
    # Right: Parameters (Consolidated)
    ax_params = axs[1]
    # Colors: mu_1 (r), mu_2 (g), P_mu_1 (r), P_mu_2 (g), P_X (k) - Solid Lines
    if filtered_data and "optimized_params" in filtered_data[0]:
        # Extract individual arrays
        mu_1 = [entry["optimized_params"]["mu_1"] for entry in filtered_data]
        mu_2 = [entry["optimized_params"]["mu_2"] for entry in filtered_data]
        P_mu_1 = [entry["optimized_params"]["P_mu_1"] for entry in filtered_data]
        P_mu_2 = [entry["optimized_params"]["P_mu_2"] for entry in filtered_data]
        P_X = [entry["optimized_params"]["P_X_value"] for entry in filtered_data]
        
        ax_params.plot(fiber_lengths, mu_1, 'r', label='mu_1')
        ax_params.plot(fiber_lengths, mu_2, 'g', label='mu_2')
        ax_params.plot(fiber_lengths, P_mu_1, 'r', label='P_mu_1')
        ax_params.plot(fiber_lengths, P_mu_2, 'g', label='P_mu_2')
        ax_params.plot(fiber_lengths, P_X, 'k', label='P_X')

    ax_params.set_xlabel("Fiber Length (km)")
    ax_params.set_ylabel("Parameter Value")
    ax_params.set_title("Optimized Parameters")
    ax_params.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax_params.set_ylim(0.0, 1.0)
    ax_params.grid(True)
    
    plt.tight_layout()
    plot_filename = os.path.join(output_dir_plots, f"FastLocalSearch_detailed_nx_{target_nx:.0e}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    print(f"📊 Saved plot to {plot_filename}")
    plt.close() # Close plot to free memory

if __name__ == '__main__':
    # Define constants and bounds
    bounds = [(4e-4, 0.9), (2e-4, 0.5), (1e-12, 1.0 - 1e-12), (1e-12, 1.0 - 1e-12), (1e-12, 1.0 - 1e-12)]

    # --- Specific Initial Guesses for each n_X ---
    initial_guesses_map = {
        1e4: np.array([0.65, 0.15, 0.05, 0.61, 0.425]),
        1e5: np.array([0.62, 0.24, 0.10, 0.70, 0.55]),
        1e6: np.array([0.68, 0.30, 0.14, 0.74, 0.66]),
        1e7: np.array([0.55, 0.34, 0.15, 0.75, 0.75]),
        1e8: np.array([0.54, 0.375, 0.16, 0.775, 0.83]),
        1e9: np.array([0.52, 0.40, 0.18, 0.785, 0.88]),
    }
    
    # Fallback default
    default_guess = np.array([0.52, 0.40, 0.18, 0.785, 0.88])

    L_values = np.linspace(0, 200, 1000)
    
    # Run for ALL values
    n_X_values = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9] 

    # Optimization parameters
    alpha = 0.2
    eta_Bob = 0.1
    P_dc_value = 6*10**-7
    epsilon_sec = 1e-10
    epsilon_cor = 1e-15
    f_EC = 1.16
    e_mis = 0.01
    P_ap = 0
    n_event = 1

    final_results = []
    
    print("🚀 Starting Optimization...")
    timing_data = {}
    total_sequential_time = 0.0
    start_time_wall_clock = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(n_X_values)) as executor:
        futures = {}
        for n_X in n_X_values:
            # Pick the specific guess for this n_X (or default if missing)
            guess_for_this_n_X = initial_guesses_map.get(n_X, default_guess)
            
            future = executor.submit(
                optimize_single_nx_sequence, 
                n_X, L_values, bounds, guess_for_this_n_X, 
                alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event
            )
            futures[future] = n_X
        
        for future in concurrent.futures.as_completed(futures):
            n_X_done = futures[future]
            try:
                batch_result, duration = future.result()
                final_results.extend(batch_result)
                total_sequential_time += duration
                timing_data[str(float(n_X_done))] = duration
                print(f"✅ Finished optimizing chain for n_X={n_X_done:.0e} in {duration:.2f}s")
            except Exception as exc:
                print(f"Generated an exception for n_X={n_X_done}: {exc}")

    wall_time_duration = time.time() - start_time_wall_clock

    # Sort results
    final_results.sort(key=lambda x: (x[1], x[0])) 

    # Generate and save dataset
    dataset = [{
        "fiber_length": float(r[0]),
        "n_X": int(r[1]),
        "key_rate": float(r[2]),
        "optimized_parameters": { "mu_1": float(r[3][0]), "mu_2": float(r[3][1]), "P_mu_1": float(r[3][2]), "P_mu_2": float(r[3][3]), "P_X_value": float(r[3][4]) },
        "initial_guess": { "mu_1": float(r[4][0]), "mu_2": float(r[4][1]), "P_mu_1": float(r[4][2]), "P_mu_2": float(r[4][3]), "P_X_value": float(r[4][4]) }
    } for r in final_results]

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # Save in current directory (Optimization/script_version)
    results_filename = f'qkd_optimization_results_{timestamp}.json'
    results_path = os.path.join(script_dir, results_filename)
    
    with open(results_path, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"\nOptimization Complete! Saved to {results_path}")
    
    # --- Post-processing: Grouping ---
    print("\n📦 Grouping results...")
    
    # Fixed parameters for calculating metrics
    P_dc_value_metric = 6e-7  
    e_mis_metric = 5e-3  

    grouped_data = {}

    for entry in dataset:
        fiber_length = entry["fiber_length"]
        n_X = float(entry["n_X"]) 

        # Compute features
        e_1 = fiber_length / 100  
        e_2 = -np.log10(P_dc_value_metric) 
        e_3 = e_mis_metric * 100  
        e_4 = np.log10(n_X)  

        formatted_entry = {
            "fiber_length": fiber_length,
            "e_1": e_1,
            "e_2": e_2,
            "e_3": e_3,
            "e_4": e_4,
            "key_rate": entry["key_rate"],
            "optimized_params": entry["optimized_parameters"], 
        }

        n_X_str = str(n_X) 
        if n_X_str not in grouped_data:
            grouped_data[n_X_str] = []
        grouped_data[n_X_str].append(formatted_entry)

    grouped_filename = f'qkd_grouped_dataset_{timestamp}.json'
    grouped_path = os.path.join(script_dir, grouped_filename)

    with open(grouped_path, 'w') as f:
        json.dump(grouped_data, f, indent=2)

    print(f"✅ Grouped dataset saved as: {grouped_path}")

    # --- Post-processing: Reordering and Saving to Good Directory ---
    print("\n🔄 Reordering and exporting...")
    
    # Reorder by using the file we just saved
    reordered_data = reorder_json_by_fiber_length(grouped_path)
    
    if reordered_data:
        # Define output directory
        # Go up one level from script_dir to Optimization, then to ../Training_Data... 
        # Actually standard path is from project root.
        # project_root/Training_Data/n_X/good
        # project_root/Training_Data/n_X/good
        output_dir = os.path.join(script_dir, "results_local")

        os.makedirs(output_dir, exist_ok=True)
        
        output_filename = f"reordered_{grouped_filename}"
        output_file_path = os.path.join(output_dir, output_filename)
        
        print(f"💾 Saving final output to: {output_file_path}")
        
        with open(output_file_path, 'w') as outfile:
            json.dump(reordered_data, outfile, indent=2)
            
        # --- Post-processing: Plotting ---
        print("\n📈 Generating plots...")
        for n_X_val in n_X_values:
            plot_for_nx(reordered_data, n_X_val, output_dir)
             
        # Generate timing report
        print("\n⏱ Generating Timing Report...")
        try:
             generate_timing_report.generate_report(
                 timing_data,
                 total_sequential_time,
                 wall_time_duration,
                 len(n_X_values), # Approx cores used
                 output_dir,
                 filename_prefix="FastLocalSearch_"
             )
        except Exception as e:
            print(f"Error generating timing report: {e}")
            
    print("\n✨ All tasks completed successfully.")
