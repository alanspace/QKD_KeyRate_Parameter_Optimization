import os
import sys
import time
import json
import logging
import numpy as np
import concurrent.futures
from scipy.optimize import dual_annealing

# --- ENVIRONMENT CONFIGURATION ---
# We explicitly do NOT import JAX here to ensure a pure NumPy environment.
# This script uses standard NumPy for all physics calculations.

# Setup Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

# Assumption: You have a NumPy-only version of your physics model.
# If src.qkd.model is JAX-based, we use a wrapper that converts JAX arrays back to NumPy
# or assumes a pure numpy function 'scalar_objective_numpy' exists.
from src.qkd.model_numpy import scalar_objective_numpy 

# --- CONFIGURATION ---
RUN_NAME = "tier1_legacy_baseline"
OUTPUT_DIR = os.path.join(os.path.dirname(script_dir), "outputs", RUN_NAME)

# Physics Bounds
# Updated to match JAX script exactly (1e-6) for fair comparison
Bounds = [
    (4e-4, 0.9),           # mu_1
    (2e-4, 0.5),           # mu_2
    (1e-6, 1.0 - 1e-6),    # P_mu_1
    (1e-6, 1.0 - 1e-6),    # P_mu_2
    (1e-6, 1.0 - 1e-6)     # P_X_value
]

# Standard Initial Guesses for the first L-point
INITIAL_GUESSES_MAP = {
    1e4: np.array([0.65, 0.15, 0.05, 0.61, 0.425]),
    1e5: np.array([0.62, 0.24, 0.10, 0.70, 0.55]),
    1e6: np.array([0.68, 0.30, 0.14, 0.74, 0.66]),
    1e7: np.array([0.55, 0.34, 0.15, 0.75, 0.75]),
    1e8: np.array([0.54, 0.375, 0.16, 0.775, 0.83]),
    1e9: np.array([0.52, 0.40, 0.18, 0.785, 0.88]),
}

def setup_logging(run_timestamp):
    log_dir = os.path.join(OUTPUT_DIR, f"run_{run_timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"tier1_optimization.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
    )
    return log_dir

def optimize_single_nx_curve_tier1(n_X, L_values, bounds, constants_tuple):
    """
    Tier 1 Baseline Logic:
    - Pure NumPy objective function.
    - No JAX gradients (uses SciPy's numerical 2-point estimation).
    - Tuned Dual Annealing (DA) for global search.
    - Continuity-based warm start (previous L seeds next L).
    """
    start_time = time.time()
    results = []
    current_x0 = INITIAL_GUESSES_MAP.get(n_X, np.array([0.52, 0.40, 0.18, 0.78, 0.88]))

    # --- TUNED DUAL ANNEALING PARAMETERS ---
    # visit: Generalized Simulated Annealing parameter (2.0-2.7 range)
    # initial_temp: Lowered from 5250 to 2500 to reduce noise/jitter
    # minimizer_kwargs: Internal L-BFGS-B (No JAX, uses numerical jac)
    DA_KWARGS = {
        'initial_temp': 2500.0,
        'visit': 2.62,
        'maxiter': 500,     # Reduced iterations for "fair" speed vs JAX
        'no_local_search': False,
        'minimizer_kwargs': {
            'method': 'L-BFGS-B',
            'options': {'ftol': 1e-10, 'gtol': 1e-10}
        }
    }

    for i, L in enumerate(L_values):
        # Objective wrapper for current L
        def wrapped_obj(params):
            # Sum of probabilities constraint (Smooth Penalty)
            # Replacing hard 1e6 with smooth quadratic penalty to help L-BFGS-B gradients
            violation = (params[2] + params[3]) - 0.999
            if violation > 0:
                return 1e6 * (1 + violation**2) 
            
            # Physics call (Negative rate for maximization)
            return scalar_objective_numpy(params, L, n_X, *constants_tuple)

        # Execute Global Optimization
        # x0 provides a "hint" to the annealing process to maintain continuity
        res = dual_annealing(
            func=wrapped_obj,
            bounds=bounds,
            x0=current_x0,
            **DA_KWARGS
        )

        rate = -res.fun
        optimized_params = res.x
        
        # Update continuity chain
        if rate > 1e-18:
            current_x0 = optimized_params

        results.append({
            "fiber_length": float(L),
            "n_X": float(n_X),
            "key_rate": float(rate) if rate > 0 else 0.0,
            "optimized_parameters": {
                "mu_1": float(optimized_params[0]),
                "mu_2": float(optimized_params[1]),
                "P_mu_1": float(optimized_params[2]),
                "P_mu_2": float(optimized_params[3]),
                "P_X_value": float(optimized_params[4])
            }
        })

        if i % 25 == 0:
            logging.info(f"n_X={n_X:.0e} | L={L:5.1f}km | Rate={rate:.2e}")

    duration = time.time() - start_time
    return n_X, results, duration

def run_tier1_pipeline():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = setup_logging(timestamp)
    
    # Grid: 1000 points (0.2km steps)
    L_values = np.linspace(0, 200, 1000)
    n_X_values = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
    
    # Physics Constants (Standard)
    constants_tuple = (
        0.2,    # alpha
        0.1,    # eta_Bob
        6e-7,   # P_dc
        1e-10,  # epsilon_sec
        1e-15,  # epsilon_cor
        1.16,   # f_EC
        0.01,   # e_mis
        0.0,    # P_ap
        1.0     # n_event
    )

    all_results = {}
    timing_data = {}
    start_wall = time.time()

    # Utilize ProcessPool to run n_X curves in parallel (the "fair" hardware usage)
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {
            executor.submit(optimize_single_nx_curve_tier1, n_X, L_values, Bounds, constants_tuple): n_X 
            for n_X in n_X_values
        }
        
        for future in concurrent.futures.as_completed(futures):
            n_X_done = futures[future]
            try:
                nx_key, res_list, duration = future.result()
                all_results[str(float(nx_key))] = res_list
                timing_data[str(float(nx_key))] = duration
                logging.info(f"✅ Completed n_X={n_X_done:.0e} in {duration/60:.2f} min")
            except Exception as e:
                logging.error(f"❌ n_X={n_X_done:.0e} failed: {e}")

    # Save Results
    json_path = os.path.join(log_dir, f"results_tier1_legacy_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    total_seq = sum(timing_data.values())
    wall_time = time.time() - start_wall
    
    logging.info("="*50)
    logging.info(f"TIER 1 SUMMARY")
    logging.info(f"Total Sequential Time: {total_seq/3600:.2f} hrs")
    logging.info(f"Wall Clock Time:      {wall_time/60:.2f} min")
    logging.info("="*50)

if __name__ == "__main__":
    run_tier1_pipeline()
