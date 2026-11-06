import os
# --- PREVENTION GUARDS (Set before JAX import) ---
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import sys
import time
import json
import logging
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
from scipy.optimize import dual_annealing, minimize
import concurrent.futures

# Setup pathing
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

# Add tools directory
tools_dir = os.path.join(os.path.dirname(script_dir), "tools")
if tools_dir not in sys.path:
    sys.path.append(tools_dir)

import generate_timing_report
import plot_results as plot_final_results

# Import JAX objective
from src.qkd.model import scalar_objective 

# Config
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(process)d] %(levelname)s - %(message)s'
)

INITIAL_GUESSES_MAP = {
    1e4: np.array([0.65, 0.15, 0.05, 0.61, 0.425]),
    1e5: np.array([0.62, 0.24, 0.10, 0.70, 0.55]),
    1e6: np.array([0.68, 0.30, 0.14, 0.74, 0.66]),
    1e7: np.array([0.55, 0.34, 0.15, 0.75, 0.75]),
    1e8: np.array([0.54, 0.375, 0.16, 0.775, 0.83]),
    1e9: np.array([0.52, 0.40, 0.18, 0.785, 0.88]),
}

DANGER_ZONE_STARTS = {
    1e9: 150.0, 
    1e8: 140.0, 
    1e7: 120.0, 
    1e6: 100.0, 
    1e5: 75.0, 
    1e4: 40.0
}

objective_val_and_grad = jit(value_and_grad(scalar_objective, argnums=0))

def optimize_adaptive_robust(L, n_X, bounds, constants, P_prev):
    """
    Gold Standard Adaptive Logic:
    1. Try Local Search (Warm Start).
    2. If it fails, use Regularized Dual Annealing (prevents jumping basins).
    """
    danger_start = DANGER_ZONE_STARTS.get(n_X, 150.0)
    
    # --- STEP 1: LOCAL SEARCH ---
    # JAX Gradient Wrapper for minimize (L-BFGS-B)
    def local_obj(p):
        p_jax = jnp.array(p)
        v, g = objective_val_and_grad(p_jax, L, n_X, *constants)
        return float(v), np.array(g, dtype=np.float64)

    best_rate = 0.0
    best_params = P_prev.copy()

    # Try local first for speed and smoothness
    try:
        res = minimize(
            local_obj, 
            x0=P_prev, 
            bounds=bounds, 
            method='L-BFGS-B', 
            jac=True,
            options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 1000}
        )
        best_rate = -res.fun
        best_params = res.x
    except: 
        pass

    # --- STEP 2: GLOBAL ESCAPE (REGULARIZED) ---
    # Trigger if Local is stuck at 0 OR if we are in deep danger
    # L2 Penalty logic: Force DA to prefer the current basin
    if best_rate < 1e-12 or L > danger_start:
        lambda_reg = 1e2 # Strength of the "leash"
        
        # Scalar objective with L2 leash for SciPy global steps
        def regularized_global_obj(p):
            p_jax = jnp.array(p)
            rate_loss = scalar_objective(p_jax, L, n_X, *constants)
            dist_penalty = lambda_reg * jnp.sum((p_jax - P_prev)**2)
            return float(rate_loss + dist_penalty)

        # Gradient with L2 leash for SciPy local searches inside DA
        def reg_grad(p):
            p_jax = jnp.array(p)
            _, g = objective_val_and_grad(p_jax, L, n_X, *constants)
            g_reg = 2 * lambda_reg * (p_jax - P_prev)
            return np.array(g + g_reg, dtype=np.float64)

        try:
            res_ga = dual_annealing(
                func=regularized_global_obj,
                bounds=bounds,
                x0=P_prev,
                initial_temp=2500,
                maxiter=500, # Faster than the standard 1000 for adaptive logic
                minimizer_kwargs={
                    'method': 'L-BFGS-B', 
                    'jac': reg_grad, 
                    'options': {'ftol': 1e-12, 'maxiter': 1000}
                }
            )
            # Re-evaluate raw physics rate at this point (without the leash)
            actual_rate = -float(scalar_objective(jnp.array(res_ga.x), L, n_X, *constants))
            
            # Switch only if Global found something better (in physicist terms)
            if actual_rate > best_rate:
                best_rate = actual_rate
                best_params = res_ga.x
        except: 
            pass

    return max(0.0, best_rate), best_params

def run_single_curve(n_X, L_values, bounds, constants):
    process_id = os.getpid()
    logging.info(f"Worker {process_id} starting robust curve n_X={n_X:.0e}")
    
    start_time = time.time()
    P_curr = INITIAL_GUESSES_MAP.get(n_X, np.array([0.52, 0.40, 0.18, 0.785, 0.88]))
    results = []
    
    for i, L in enumerate(L_values):
        rate, P_curr = optimize_adaptive_robust(L, n_X, bounds, constants, P_curr)
        results.append({
            "fiber_length": float(L), 
            "n_X": int(n_X), 
            "key_rate": float(rate),
            "optimized_parameters": {
                "mu_1": float(P_curr[0]), "mu_2": float(P_curr[1]),
                "P_mu_1": float(P_curr[2]), "P_mu_2": float(P_curr[3]),
                "P_X_value": float(P_curr[4])
            }
        })
        if i % 100 == 0: 
            logging.info(f"Worker {process_id} | n_X={n_X:.0e} | L={L:.1f} | Rate={rate:.2e}")
            
    duration = time.time() - start_time
    logging.info(f"✅ Worker {process_id} finished n_X={n_X:.0e} in {duration:.2f}s")
    return n_X, results, duration

def run_pipeline():
    # Setup
    base_dir = os.path.join(os.path.dirname(script_dir), "outputs", "tier3_robust_adaptive_jax")
    os.makedirs(base_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Constants
    bounds = [(4e-4, 0.9), (2e-4, 0.5), (1e-6, 1.0-1e-6), (1e-6, 1.0-1e-6), (1e-6, 1.0-1e-6)]
    L_values = np.linspace(0, 200, 1000) 
    n_X_values = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
    constants = (0.2, 0.1, 6e-7, 1e-10, 1e-15, 1.16, 0.01, 0, 1)

    final_results = []
    timing_per_nx = {}
    total_time_sequential = 0.0
    start_wall = time.time()
    
    # Parallel execution
    max_workers = 6 
    logging.info(f"Starting Robust Adaptive Optimization in PARALLEL mode ({max_workers} workers)...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_single_curve, n_X, L_values, bounds, constants): n_X 
            for n_X in n_X_values
        }
        
        for future in concurrent.futures.as_completed(futures):
            n_X_done = futures[future]
            try:
                nx_key, res, dur = future.result()
                final_results.extend(res)
                timing_per_nx[str(float(nx_key))] = dur
                total_time_sequential += dur
            except Exception as e:
                logging.error(f"n_X={n_X_done} failed: {e}")
                import traceback
                logging.error(traceback.format_exc())

    # Save
    json_path = os.path.join(run_dir, f"robust_results_jax_{timestamp}.json")
    grouped = {}
    for r in final_results:
        k = str(float(r['n_X']))
        if k not in grouped: grouped[k] = []
        grouped[k].append(r)
    
    with open(json_path, 'w') as f: 
        json.dump(grouped, f, indent=2)
    
    # Reporting
    wall_duration = time.time() - start_wall
    try:
        generate_timing_report.generate_report(
            timing_per_nx, total_time_sequential, wall_duration, max_workers, run_dir, "RobustJAX_"
        )
        plot_final_results.generate_plots(json_path, "RobustJAX_")
        logging.info(f"✅ Success! Results and plots saved to {run_dir}")
    except Exception as e:
        logging.error(f"Reporting error: {e}")

if __name__ == "__main__":
    run_pipeline()
