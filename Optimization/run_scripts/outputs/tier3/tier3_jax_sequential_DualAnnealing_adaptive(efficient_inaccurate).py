import os
# --- PREVENTION GUARDS (Set before JAX import) ---
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
os.environ["OMP_NUM_THREADS"] = "2"  # Limit each worker threads to prevent thundering herd
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

# Setup path
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
DEFAULT_GUESS = np.array([0.52, 0.40, 0.18, 0.785, 0.88])

DANGER_ZONE_STARTS = {
    1e9: 150.0,
    1e8: 140.0,
    1e7: 120.0,
    1e6: 100.0,
    1e5: 75.0,
    1e4: 40.0,
}

LOCAL_KWARGS = {
    'method': 'L-BFGS-B',
    'jac': True,
    'options': {'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 1500}
}

DA_KWARGS = {
    'initial_temp': 2500.0, 
    'visit': 2.62,          
    'maxiter': 1000,        
    'no_local_search': False,
    'minimizer_kwargs': {
        'method': 'L-BFGS-B',
        'jac': True,
        'options': {'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 1000}
    }
}

objective_val_and_grad = jit(value_and_grad(scalar_objective, argnums=0))

def get_scipy_funcs(L, n_X, constants):
    def scipy_val(p):
        p_jax = jnp.array(p)
        val = scalar_objective(p_jax, L, n_X, *constants)
        return float(val)

    def scipy_grad(p):
        p_jax = jnp.array(p)
        _, grad = objective_val_and_grad(p_jax, L, n_X, *constants)
        return np.array(grad, dtype=np.float64)
    
    def scipy_both(p):
        p_jax = jnp.array(p)
        v, g = objective_val_and_grad(p_jax, L, n_X, *constants)
        return float(v), np.array(g, dtype=np.float64)

    return scipy_val, scipy_grad, scipy_both

def optimize_adaptive(L, n_X, bounds, constants, initial_guess=None):
    scipy_val, scipy_grad, scipy_both = get_scipy_funcs(L, n_X, constants)
    danger_threshold = DANGER_ZONE_STARTS.get(n_X, 150.0)
    best_rate = 0.0
    best_params = initial_guess if initial_guess is not None else DEFAULT_GUESS

    if L < danger_threshold + 10.0:
        try:
            res = minimize(fun=scipy_both, x0=best_params, bounds=bounds, **LOCAL_KWARGS)
            best_rate = -res.fun
            best_params = res.x
        except: pass

    if best_rate < 1e-12:
        kwargs = DA_KWARGS.copy()
        kwargs['minimizer_kwargs']['jac'] = scipy_grad
        if initial_guess is not None: kwargs['x0'] = initial_guess
        try:
            res = dual_annealing(func=scipy_val, bounds=bounds, **kwargs)
            if -res.fun > best_rate:
                best_rate = -res.fun
                best_params = res.x
        except Exception as e:
            logging.warning(f"DA failed at L={L}: {e}")

    if best_rate < 1e-15: best_rate = 0.0
    return best_rate, best_params

def run_single_nx_curve(n_X, L_values, bounds, constants):
    process_id = os.getpid()
    logging.info(f"Worker {process_id} starting curve n_X={n_X:.0e}")
    start_time = time.time()
    results = []
    P_prev = INITIAL_GUESSES_MAP.get(n_X, DEFAULT_GUESS)
    
    for i, L in enumerate(L_values):
        if i % 100 == 0:
            logging.info(f"Worker {process_id} | n_X={n_X:.0e} | L={L:.1f}")
        
        rate, params = optimize_adaptive(L, n_X, bounds, constants, initial_guess=P_prev)
        if rate > 1e-18: P_prev = params.copy()
            
        results.append({
            "fiber_length": float(L), "n_X": int(n_X), "key_rate": float(rate),
            "optimized_parameters": {
                "mu_1": float(params[0]), "mu_2": float(params[1]), 
                "P_mu_1": float(params[2]), "P_mu_2": float(params[3]), 
                "P_X_value": float(params[4])
            }
        })
        
    duration = time.time() - start_time
    logging.info(f"✅ Worker {process_id} finished n_X={n_X:.0e} in {duration:.2f}s")
    return n_X, results, duration

def run_jax_pipeline():
    base_dir = os.path.join(os.path.dirname(script_dir), "outputs", "tier3_parallel_adaptive_jax")
    os.makedirs(base_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    bounds = [(4e-4, 0.9), (2e-4, 0.5), (1e-6, 1.0-1e-6), (1e-6, 1.0-1e-6), (1e-6, 1.0-1e-6)]
    L_values = np.linspace(0, 200, 1000) 
    n_X_values = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
    constants = (0.2, 0.1, 6e-7, 1e-10, 1e-15, 1.16, 0.01, 0, 1)

    final_results = []
    timing_per_nx = {}
    total_time = 0.0
    start_wall = time.time()
    
    # PARALLEL EXECUTION (12 CORES) - RE-ATTEMPTING WITH GUARDS
    max_workers = 12
    logging.info(f"Starting optimization in PARALLEL mode ({max_workers} workers) with THREAD GUARDS...")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_single_nx_curve, n_X, L_values, bounds, constants): n_X 
            for n_X in n_X_values
        }
        
        for future in concurrent.futures.as_completed(futures):
            n_X_done = futures[future]
            try:
                _, res, dur = future.result()
                final_results.extend(res)
                timing_per_nx[str(float(n_X_done))] = dur
                total_time += dur
            except Exception as e:
                logging.error(f"n_X={n_X_done} failed: {e}")
                import traceback
                logging.error(traceback.format_exc())

    json_path = os.path.join(run_dir, f"parallel_qkd_results_jax_{timestamp}.json")
    grouped = {}
    for r in final_results:
        k = str(float(r['n_X']))
        if k not in grouped: grouped[k] = []
        grouped[k].append(r)
    
    with open(json_path, 'w') as f: json.dump(grouped, f, indent=2)
    
    wall_duration = time.time() - start_wall
    try:
        generate_timing_report.generate_report(timing_per_nx, total_time, wall_duration, max_workers, run_dir, "ParallelJAX_")
        plot_final_results.generate_plots(json_path, "ParallelJAX_")
    except Exception as e:
        print(f"Reporting error: {e}")

if __name__ == "__main__":
    run_jax_pipeline()
