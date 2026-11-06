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
from jax import jit, value_and_grad, hessian
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

# --- GENERATION 4 JAX ENGINE ---
# We use value_and_grad for 1st order and a separate hessian for 2nd order
# static_argnums is used because L and n_X are constants for a single optimization call
objective_vg = jit(value_and_grad(scalar_objective, argnums=0))
objective_hess = jit(hessian(scalar_objective, argnums=0))

def optimize_gen4(L, n_X, bounds, constants, P_prev, beta=1e-4): # Ultra-loose leash to allow true peak hunting
    """
    Generation 4: Hessian-Accelerated Physical Basin Tracking.
    1. Newton-Method Refinement (with Hessian).
    2. Fallback to Regularized Dual Annealing (with Leash).
    """
    danger_start = DANGER_ZONE_STARTS.get(n_X, 150.0)
    
    # --- STEP 1: MULTI-OBJECTIVE LOCAL SEARCH (Hessian-based) ---
    # We incorporate the "Physical Controller" (beta penalty) directly into the local step
    def local_objective_full(p):
        p_jax = jnp.array(p)
        # Physics Loss
        val_raw, grad_raw = objective_vg(p_jax, L, n_X, *constants)
        
        # Continuity Penalty (Leash)
        diff = p_jax - P_prev
        penalty = beta * jnp.sum(diff**2)
        penalty_grad = 2 * beta * diff
        
        return float(val_raw + penalty), np.array(grad_raw + penalty_grad, dtype=np.float64)

    def local_hessian_full(p):
        p_jax = jnp.array(p)
        # Raw Physics Hessian
        h_raw = objective_hess(p_jax, L, n_X, *constants)
        # Penalty Hessian is just diagonal matrix of 2 * beta
        h_penalty = jnp.eye(5) * (2 * beta)
        return np.array(h_raw + h_penalty, dtype=np.float64)

    best_rate = 0.0
    best_params = P_prev.copy()

    # Refinement using trust-region which utilizes 2nd order data
    try:
        res = minimize(
            local_objective_full,
            x0=P_prev,
            method='trust-constr', # Native support for Hessian + Bounds
            jac=True,
            hess=local_hessian_full,
            bounds=bounds,
            options={'gtol': 1e-10, 'xtol': 1e-10, 'maxiter': 500}
        )
        # Evaluate raw physics (without the anchor)
        rate_found = -float(scalar_objective(jnp.array(res.x), L, n_X, *constants))
        best_rate = max(0.0, rate_found)
        best_params = res.x
    except Exception as e:
        # Fallback to L-BFGS-B if trust-region fails (numerical stability)
        try:
            res_lb = minimize(local_objective_full, x0=P_prev, bounds=bounds, method='L-BFGS-B', jac=True)
            rate_found = -float(scalar_objective(jnp.array(res_lb.x), L, n_X, *constants))
            best_rate = max(0.0, rate_found)
            best_params = res_lb.x
        except: pass

    # --- STEP 2: GLOBAL ESCAPE IF NEEDED ---
    # Trigger if rate is near zero or in danger zone with poor local result
    if (best_rate < 1e-12 and L > 5.0) or (L > danger_start and best_rate < 1e-15):
        try:
            res_ga = dual_annealing(
                func=lambda p: float(scalar_objective(jnp.array(p), L, n_X, *constants) + beta*jnp.sum((jnp.array(p)-P_prev)**2)),
                bounds=bounds,
                x0=P_prev,
                maxiter=400,
                minimizer_kwargs={'method': 'L-BFGS-B', 'jac': True, 'options': {'ftol': 1e-12}}
            )
            rate_ga = -float(scalar_objective(jnp.array(res_ga.x), L, n_X, *constants))
            if rate_ga > best_rate:
                best_rate = rate_ga
                best_params = res_ga.x
        except: pass

    return best_rate, best_params

def run_single_curve_gen4(n_X, bounds, constants):
    process_id = os.getpid()
    logging.info(f"Worker {process_id} starting GEN-4 curve n_X={n_X:.0e} (Mixed Resolution)")
    
    # --- MIXED RESOLUTION PLAN ---
    # 0 to 120 km: 2.0 km steps (Stable)
    # 120 to 150 km: 0.5 km steps (Transition)
    # 150 to 200 km: 0.1 km steps (Hard Danger Zone)
    # --- ULTRA HIGH RESOLUTION MIXED GRID ---
    # Capturing every subtle physical transition
    L_values = np.concatenate([
        np.arange(0, 100, 0.5),      # 0.5km steps (Stable)
        np.arange(100, 150, 0.1),    # 0.1km steps (Transition)
        np.arange(150, 200.02, 0.02) # 0.02km steps (Extreme subtlety in Danger Zone)
    ])
    
    start_time = time.time()
    P_curr = INITIAL_GUESSES_MAP.get(n_X, np.array([0.52, 0.40, 0.18, 0.785, 0.88]))
    results = []
    
    for i, L in enumerate(L_values):
        # Adaptive Beta: Kept ultra-low to allow the optimizer to "win" against the baseline
        beta_curr = 1e-3 if L > 150 else 1e-4
        
        rate, P_curr = optimize_gen4(L, n_X, bounds, constants, P_curr, beta=beta_curr)
        
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
    logging.info(f"✅ Worker {process_id} finished GEN-4 n_X={n_X:.0e} in {duration:.2f}s")
    return n_X, results, duration

def run_pipeline_gen4():
    base_dir = os.path.join(os.path.dirname(script_dir), "outputs", "tier3_generation4_jax")
    os.makedirs(base_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    bounds = [(4e-4, 0.9), (2e-4, 0.5), (1e-6, 1.0-1e-6), (1e-6, 1.0-1e-6), (1e-6, 1.0-1e-6)]
    n_X_values = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
    constants = (0.2, 0.1, 6e-7, 1e-10, 1e-15, 1.16, 0.01, 0, 1)

    final_results = []
    timing_per_nx = {}
    total_time_sequential = 0.0
    start_wall = time.time()
    
    max_workers = 6 
    logging.info(f"🚀 Launching Generation 4 Optimization (Parallel-6)...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_single_curve_gen4, n_X, bounds, constants): n_X for n_X in n_X_values}
        for future in concurrent.futures.as_completed(futures):
            n_x_key, res, dur = future.result()
            final_results.extend(res)
            timing_per_nx[str(float(n_x_key))] = dur
            total_time_sequential += dur

    # Save and Report
    json_path = os.path.join(run_dir, f"gen4_results_jax_{timestamp}.json")
    grouped = {str(float(nx)): [r for r in final_results if r['n_X'] == nx] for nx in n_X_values}
    with open(json_path, 'w') as f: json.dump(grouped, f, indent=2)
    
    wall_duration = time.time() - start_wall
    generate_timing_report.generate_report(timing_per_nx, total_time_sequential, wall_duration, max_workers, run_dir, "Gen4_")
    plot_final_results.generate_plots(json_path, "Gen4_")
    logging.info(f"✅ FINAL GEN-4 SUCCESS! Path: {run_dir}")

if __name__ == "__main__":
    run_pipeline_gen4()
