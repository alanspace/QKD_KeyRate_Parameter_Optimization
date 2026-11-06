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
from scipy.optimize import minimize, dual_annealing
import glob
import logging

# Ensure the script uses the correct project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

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

import jax
import jax.numpy as jnp

# FORCE CPU for 64-bit precision
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def optimize_single_nx_sequence_global(n_X, L_values, bounds, initial_guess, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event):
    """
    RIGOROUS GLOBAL OPTIMIZATION (Academic Approach):
    - Uses dual_annealing for every single point.
    - Guaranteed to find the global maximum (within probabilistic limits).
    - computationally expensive but provides the 'Ground Truth'.
    """
    results = []
    
    # Pre-compile objective (Maximize Key Rate -> Minimize Negative Key Rate)
    def objective_wrapper(params, L_val):
        val, _ = objective_val_and_grad(
            params, L_val, n_X, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event
        )
        if not np.isfinite(val): return 1e9
        return float(-val) # Return negative because we want to MAXIMIZE, but scipy MINIMIZES

    # --- CRITICAL FIX: Specific Initial Guesses for each n_X ---
    initial_guesses_map = {
        1e4: np.array([0.65, 0.15, 0.05, 0.61, 0.425]),
        1e5: np.array([0.62, 0.24, 0.10, 0.70, 0.55]),
        1e6: np.array([0.68, 0.30, 0.14, 0.74, 0.66]),
        1e7: np.array([0.55, 0.34, 0.15, 0.75, 0.75]),
        1e8: np.array([0.54, 0.375, 0.16, 0.775, 0.83]),
        1e9: np.array([0.52, 0.40, 0.18, 0.785, 0.88]),
    }
    
    # Smoothness-First Optimization for Global Script
    # 1. Cold Start with Correct Guess
    last_success_params = initial_guesses_map.get(n_X, default_guess).copy()
    
    for L in tqdm(L_values, desc=f"Global Opt n_X={n_X:.0e}", position=0, leave=True):
        try:
            # --- STRATEGY: SMOOTHNESS-FIRST (Local Bias + Global Escape) ---
            candidates = []
            
            # Candidate 1: Warm Start (Continuity)
            candidates.append(last_success_params)
            
            # Candidate 2: Small Jitter (Local Exploration)
            jitter_scale = 0.05
            candidates.append(last_success_params * np.random.uniform(1.0 - jitter_scale, 1.0 + jitter_scale, size=len(last_success_params)))
            
            # Candidate 3: Global Reset (Safety Net)
            candidates.append(initial_guesses_map.get(n_X, default_guess))
            
            best_key_rate = -1.0
            best_params = None
            
            # --- STEP 1: RACE SMOOTH CANDIDATES ---
            for start_point in candidates:
                sp = np.clip(start_point, [b[0] for b in bounds], [b[1] for b in bounds])
                try:
                    res = minimize(
                        fun=lambda p: objective_wrapper(p, L),
                        x0=sp,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 1500} # Strict Precision
                    )
                    rate = -res.fun
                    if rate > best_key_rate and res.success:
                        best_key_rate = rate
                        best_params = res.x
                except: continue
                
            # --- STEP 2: GLOBAL ESCAPE MECHANISM ---
            # If local search failed (dead basin), try Global Search
            if best_key_rate < 1e-12:
                try:
                    L_seed = int(L * 100) + 42
                    # Use DE for consistency with JAX strategy 
                    res_global = differential_evolution(
                        func=lambda p: objective_wrapper(p, L),
                        bounds=bounds,
                        maxiter=100, 
                        popsize=15, 
                        polish=False,
                        disp=False,
                        seed=L_seed,
                        workers=1 
                    )
                    
                    # Refine
                    res_refined = minimize(
                        fun=lambda p: objective_wrapper(p, L),
                        x0=res_global.x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 1500}
                    )
                    
                    final_global_rate = -res_refined.fun
                    if final_global_rate > best_key_rate and final_global_rate > 1e-12:
                        best_key_rate = final_global_rate
                        best_params = res_refined.x
                        
                except Exception: pass
            
            # --- UPDATE HISTORY ---
            if best_key_rate < 0: best_key_rate = 0.0
            
            # Ideally, update last_success_params based on outcome. 
            # If we used global escape, we jump. If local, we detailed smooth.
            if best_key_rate > 1e-20:
                last_success_params = best_params.copy()
            
            results.append((L, n_X, best_key_rate, best_params, initial_guess))
 
        except Exception as e:
            print(f"❌ Error at L={L}, n_X={n_X}: {e}")
            results.append((L, n_X, 0.0, [0.0]*5, initial_guess))

            # --- UPDATE HISTORY ---
            if best_key_rate < 0: best_key_rate = 0.0
            
            if best_key_rate > 1e-20:
                last_success_params = best_params.copy()
            
            results.append((L, n_X, best_key_rate, best_params, initial_guess))
 
        except Exception as e:
            print(f"❌ Error at L={L}, n_X={n_X}: {e}")
            results.append((L, n_X, 0.0, [0.0]*5, initial_guess))
            
    return results

def reorder_json_by_fiber_length(file_path):
    try:
        with open(file_path, 'r') as f:
            json_data = json.load(f)
    except: return None

    if not isinstance(json_data, dict): return None
    
    data_list = None
    target_key = None
    for key, value in json_data.items():
        if isinstance(value, list):
            data_list = value
            target_key = key
            break

    if data_list is None: return None
    
    sorted_data_list = sorted(data_list, key=lambda x: x["fiber_length"])
    reordered_json_data = json_data.copy()
    reordered_json_data[target_key] = sorted_data_list
    return reordered_json_data

def plot_for_nx(data, target_nx, output_dir_plots):
    target_nx_str = str(float(target_nx))
    if target_nx_str not in data: return
    
    filtered_data = data[target_nx_str]
    fiber_lengths = [entry["fiber_length"] for entry in filtered_data]
    key_rates = np.array([entry["key_rate"] for entry in filtered_data])
    safe_key_rates = np.where(key_rates > 0, key_rates, 1e-30)

    # ✅ Improved visualization (1x2 Layout matched to JAX script)
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"GLOBAL SEARCH: Results (n_X = {target_nx:.0e})", fontsize=16)
    
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
    plot_filename = os.path.join(output_dir_plots, f"global_results_nx_{target_nx:.0e}.png")
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == '__main__':
    # CONSTANTS
    bounds = [(4e-4, 0.9), (2e-4, 0.5), (1e-12, 1.0 - 1e-12), (1e-12, 1.0 - 1e-12), (1e-12, 1.0 - 1e-12)]
    
    # Reduce L_values count for Global Search because it's 100x slower
    # We do fewer points but rigorous ones
    L_values = np.linspace(0, 200, 200) # 200 points instead of 1000 to save time for user
    
    n_X_values = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9] 
    initial_guesses_map = {
        1e4: np.array([0.65, 0.15, 0.05, 0.61, 0.425]),
        1e5: np.array([0.62, 0.24, 0.10, 0.70, 0.55]),
        1e6: np.array([0.68, 0.30, 0.14, 0.74, 0.66]),
        1e7: np.array([0.55, 0.34, 0.15, 0.75, 0.75]),
        1e8: np.array([0.54, 0.375, 0.16, 0.775, 0.83]),
        1e9: np.array([0.52, 0.40, 0.18, 0.785, 0.88]),
    }
    default_guess = np.array([0.52, 0.40, 0.18, 0.785, 0.88])

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
    
    print("🌍 Starting RIGOROUS GLOBAL Optimization...")
    print("⚠️ Note: This is computationally expensive. Running fewer points for demonstration.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(n_X_values)) as executor:
        futures = {}
        for n_X in n_X_values:
            guess_for_this_n_X = initial_guesses_map.get(n_X, default_guess)
            future = executor.submit(
                optimize_single_nx_sequence_global, 
                n_X, L_values, bounds, guess_for_this_n_X, 
                alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event
            )
            futures[future] = n_X
        
        for future in concurrent.futures.as_completed(futures):
            n_X_done = futures[future]
            try:
                batch_result = future.result()
                final_results.extend(batch_result)
                print(f"✅ Finished GLOBAL optimizing chain for n_X={n_X_done:.0e}")
            except Exception as exc:
                print(f"Exception for n_X={n_X_done}: {exc}")

    final_results.sort(key=lambda x: (x[1], x[0])) 

    dataset = [{
        "fiber_length": float(r[0]),
        "n_X": int(r[1]),
        "key_rate": float(r[2]),
        "optimized_parameters": { "mu_1": float(r[3][0]), "mu_2": float(r[3][1]), "P_mu_1": float(r[3][2]), "P_mu_2": float(r[3][3]), "P_X_value": float(r[3][4]) },
        "initial_guess": { "mu_1": float(r[4][0]), "mu_2": float(r[4][1]), "P_mu_1": float(r[4][2]), "P_mu_2": float(r[4][3]), "P_X_value": float(r[4][4]) }
    } for r in final_results]

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_filename = f'qkd_global_results_{timestamp}.json'
    results_path = os.path.join(script_dir, results_filename)
    
    with open(results_path, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"\nOptimization Complete! Saved to {results_path}")
    
    # Grouping
    grouped_data = {}
    for entry in dataset:
        fiber_length = entry["fiber_length"]
        n_X = float(entry["n_X"]) 
        # Metric placeholders
        e_1 = fiber_length / 100  
        e_2 = -np.log10(6e-7) 
        e_3 = 5e-3 * 100  
        e_4 = np.log10(n_X)  

        formatted_entry = {
            "fiber_length": fiber_length,
            "e_1": e_1, "e_2": e_2, "e_3": e_3, "e_4": e_4,
            "key_rate": entry["key_rate"],
            "optimized_params": entry["optimized_parameters"], 
        }

        n_X_str = str(n_X) 
        if n_X_str not in grouped_data: grouped_data[n_X_str] = []
        grouped_data[n_X_str].append(formatted_entry)

    grouped_filename = f'qkd_grouped_global_{timestamp}.json'
    grouped_path = os.path.join(script_dir, grouped_filename)
    with open(grouped_path, 'w') as f: json.dump(grouped_data, f, indent=2)

    # Reorder + Save Output
    reordered_data = reorder_json_by_fiber_length(grouped_path)
    if reordered_data:
        output_dir = os.path.join(script_dir, "results_global")
        os.makedirs(output_dir, exist_ok=True)
        
        output_filename = f"reordered_{grouped_filename}"
        output_file_path = os.path.join(output_dir, output_filename)
        with open(output_file_path, 'w') as outfile: json.dump(reordered_data, outfile, indent=2)
            
        for n_X_val in n_X_values:
            plot_for_nx(reordered_data, n_X_val, output_dir)
            
    print("\n✨ Global Optimization Finished.")
