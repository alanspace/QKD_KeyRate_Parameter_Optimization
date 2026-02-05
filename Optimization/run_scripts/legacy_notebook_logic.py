import os
import sys
import time
import json
import logging
import numpy as np
import concurrent.futures
from scipy.optimize import minimize

# --- JAX CONFIGURATION (MUST BE FIRST) ---
# Enforce CPU to ensure consistent F64 precision and avoid Metal/GPU warnings/inconsistencies
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

# Setup Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import Physics
from src.qkd.model import objective_val_and_grad

# Import Reporting Tools
import plot_final_results
import generate_timing_report

def optimize_single_nx_sequence_notebook_logic(n_X, L_values, bounds, initial_guess, constants):
    """
    The exact logic from the JAX Notebook:
    - Fast sequence for L < 100km.
    - Deep search (20+ candidates) for L >= 100km ("Danger Zone").
    """
    alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event = constants
    
    start_time = time.time()
    results = []
    current_best_guess = initial_guess.copy()
    
    # Pre-compile objective wrapper
    def objective_wrapper(params, L_val):
        val, grad = objective_val_and_grad(
            params, L_val, n_X, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event
        )
        if not np.isfinite(val): return 1e9, np.zeros_like(params)
        grad = np.clip(np.array(grad), -1e6, 1e6)
        return float(val), np.array(grad)

    total_points = len(L_values)
    for i, L in enumerate(L_values):
        if i % (max(1, total_points // 10)) == 0:
             print(f"   ... n_X={n_X:.0e} processing L={L:.1f}km ({i}/{total_points})")
             
        try:
            candidates = []
            
            # --- BASELINE CANDIDATES (Always run) ---
            candidates.append(current_best_guess) # Warm start
            candidates.append(initial_guess)      # Reset
            candidates.append(current_best_guess * np.random.uniform(0.99, 1.01, size=len(initial_guess))) # Tiny jitter
            
            # --- ADAPTIVE DEPTH ---
            if L > 100:
                # Add 20 diverse candidates to explore every nook and cranny
                for _ in range(20):
                     # Mix of small, medium, and large perturbations
                    scale = np.random.choice([0.05, 0.1, 0.2, 0.3]) # 5% to 30% jitter
                    perturbation = np.random.uniform(1.0 - scale, 1.0 + scale, size=len(initial_guess))
                    candidates.append(current_best_guess * perturbation)
                    
                # Also try random jumps from the global default
                candidates.append(initial_guess * np.random.uniform(0.8, 1.2, size=len(initial_guess)))
            else:
                # Routine checks for smooth regions
                candidates.append(current_best_guess * np.random.uniform(0.95, 1.05, size=len(initial_guess)))

            best_key_rate = -1.0
            best_params = current_best_guess
            
            # Race candidates
            for start_point in candidates:
                # Clip start point to bounds
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
                if rate > best_key_rate:
                    best_key_rate = rate
                    best_params = res.x

            if best_key_rate > 1e-20: # Threshold for valid key rate
                current_best_guess = best_params.copy()
            else:
                best_key_rate = 0.0
            
            # Formulate result dictionary matching the standard format
            result_dict = {
                "fiber_length": float(L),
                "n_X": int(n_X),
                "key_rate": float(best_key_rate),
                "optimized_parameters": {
                    "mu_1": float(best_params[0]), "mu_2": float(best_params[1]), 
                    "P_mu_1": float(best_params[2]), "P_mu_2": float(best_params[3]), 
                    "P_X_value": float(best_params[4])
                },
                "initial_guess": {}
            }
            results.append(result_dict)

        except Exception as e:
            print(f"❌ Error at L={L}, n_X={n_X}: {e}")
            # Append zero result
            results.append({
                "fiber_length": float(L), "n_X": int(n_X), "key_rate": 0.0,
                "optimized_parameters": {"mu_1":0,"mu_2":0,"P_mu_1":0,"P_mu_2":0,"P_X_value":0},
                "initial_guess": {}
            })
            
    duration = time.time() - start_time
    return n_X, results, duration

def run_notebook_pipeline():
    # --- SETUP OUTPUT FOLDER ---
    script_dir_out = os.path.dirname(os.path.abspath(__file__))
    base_output_dir = os.path.join(script_dir_out, "results_notebook_logic")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # --- LOGGING ---
    log_file = os.path.join(run_dir, "optimization_notebook.log")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(log_file, mode='w'), logging.StreamHandler(sys.stdout)])
    
    logging.info(f"🚀 Starting Notebook-Logic Optimization. Logs at {log_file}")
    logging.info(f"📂 Run Directory: {run_dir}")
    
    # --- PARAMETERS ---
    bounds = [(4e-4, 0.9), (2e-4, 0.5), (1e-6, 1.0 - 1e-6), (1e-6, 1.0 - 1e-6), (1e-6, 1.0 - 1e-6)]
    L_values = np.linspace(0, 200, 1000)
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
    
    # Initial Guesses
    initial_guesses_map = {
        1e4: np.array([0.65, 0.15, 0.05, 0.61, 0.425]),
        1e5: np.array([0.62, 0.24, 0.10, 0.70, 0.55]),
        1e6: np.array([0.68, 0.30, 0.14, 0.74, 0.66]),
        1e7: np.array([0.55, 0.34, 0.15, 0.75, 0.75]),
        1e8: np.array([0.54, 0.375, 0.16, 0.775, 0.83]),
        1e9: np.array([0.52, 0.40, 0.18, 0.785, 0.88]),
    }
    default_guess = np.array([0.52, 0.40, 0.18, 0.785, 0.88])
    
    # --- PARALLEL EXECUTION ---
    # Using ProcessPoolExecutor for true parallelism
    max_workers = os.cpu_count()
    logging.info(f"Starting parallel optimization with {max_workers} workers.")
    
    final_results = []
    total_time = 0.0
    timing_per_nx = {}
    
    start_time_wall = time.time()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_nx = {}
        for n_X in n_X_values:
            guess = initial_guesses_map.get(n_X, default_guess)
            future = executor.submit(optimize_single_nx_sequence_notebook_logic, n_X, L_values, bounds, guess, constants)
            future_to_nx[future] = n_X
            
        for future in concurrent.futures.as_completed(future_to_nx):
            n_X_done = future_to_nx[future]
            try:
                n_X_res, batch_results, duration = future.result()
                final_results.extend(batch_results)
                total_time += duration
                timing_per_nx[str(float(n_X_res))] = duration
                logging.info(f"✅ Finished n_X={n_X_res:.0e} in {duration:.2f}s")
            except Exception as e:
                logging.error(f"❌ n_X={n_X_done:.0e} failed: {e}")
                
    wall_duration = time.time() - start_time_wall
    logging.info(f"🎯 Total optimization finished in {wall_duration:.2f}s")
    
    # --- SAVE RESULTS ---
    filename = f"qkd_results_notebook_logic_{timestamp}.json"
    filepath = os.path.join(run_dir, "reordered_" + filename)
    
    grouped_data = {}
    for entry in final_results:
        nx = str(float(entry["n_X"]))
        if nx not in grouped_data: grouped_data[nx] = []
        grouped_data[nx].append(entry)
        
    with open(filepath, 'w') as f:
        json.dump(grouped_data, f, indent=2)
    logging.info(f"✅ Saved results to {filepath}")
    
    # --- REPORTING & PLOTTING ---
    logging.info("📊 Generating Plots and Reports...")
    
    # 1. Timing Report
    try:
        generate_timing_report.generate_report(
            timing_data=timing_per_nx,
            total_sequential=total_time,
            wall_time=wall_duration,
            num_cores=max_workers,
            output_dir=run_dir 
        )
    except Exception as e:
        print(f"⚠️ Error generating timing report: {e}")
        
    # 2. Plots
    try:
        plot_final_results.generate_plots(filepath)
    except Exception as e:
        print(f"⚠️ Error generating final plots: {e}")

if __name__ == "__main__":
    run_notebook_pipeline()
