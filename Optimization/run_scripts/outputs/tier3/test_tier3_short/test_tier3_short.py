import os
import sys
import time
import json
import logging
import numpy as np
import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
from scipy.optimize import dual_annealing

# Setup path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

# Add tools directory to path
tools_dir = os.path.join(os.path.dirname(script_dir), "tools")
if tools_dir not in sys.path:
    sys.path.append(tools_dir)

import generate_timing_report
import plot_results as plot_final_results

# Import JAX objective (which now includes the penalty)
from src.qkd.model import scalar_objective 

# Config
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INITIAL_GUESSES_MAP = {
    1e4: np.array([0.65, 0.15, 0.05, 0.61, 0.425]),
}
DEFAULT_GUESS = np.array([0.52, 0.40, 0.18, 0.785, 0.88])

# Tuned DA Parameters (Matched to Tier 1)
DA_KWARGS = {
    'initial_temp': 2500.0, 
    'visit': 2.62,          
    'maxiter': 1000,        
    'no_local_search': False,
    'minimizer_kwargs': {
        'method': 'L-BFGS-B',
        'jac': True,  # Using JAX gradients!
        'options': {'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 1000}
    }
}

# JIT-compile value_and_grad once
objective_val_and_grad = jit(value_and_grad(scalar_objective, argnums=0))

def optimize_dual_annealing_jax(L, n_X, bounds, constants, initial_guess=None):
    # Wrapper for SciPy (Value only for Dual Annealing global steps)
    def scipy_val(p):
        p_jax = jnp.array(p)
        val = scalar_objective(p_jax, L, n_X, *constants)
        return float(val)

    # Wrapper for Jacobian (for L-BFGS-B local search)
    def scipy_grad(p):
        p_jax = jnp.array(p)
        _, grad = objective_val_and_grad(p_jax, L, n_X, *constants)
        return np.array(grad, dtype=np.float64)

    kwargs = DA_KWARGS.copy()
    if initial_guess is not None:
        kwargs['x0'] = initial_guess
    
    kwargs['minimizer_kwargs']['jac'] = scipy_grad

    try:
        res = dual_annealing(
            func=scipy_val,
            bounds=bounds,
            **kwargs
        )
        rate = -res.fun # Flip back to positive
        best_params = res.x
        if rate < 1e-15: rate = 0.0
    except Exception as e:
        import traceback
        logging.error(f"Optimization failed at L={L}, n_X={n_X}: {e}\n{traceback.format_exc()}")
        rate = 0.0
        best_params = initial_guess if initial_guess is not None else np.zeros(5)

    return rate, best_params

def run_single_nx_curve(n_X, L_values, bounds, constants):
    start_time = time.time()
    results = []
    
    # Warm start
    P_prev = INITIAL_GUESSES_MAP.get(n_X, DEFAULT_GUESS)
    
    for i, L in enumerate(L_values):
        # Print every step for the test to see progress
        print(f"   ... Processing n_X={n_X:.0e}, L={L:.1f} ({i+1}/{len(L_values)})")
        
        rate, params = optimize_dual_annealing_jax(L, n_X, bounds, constants, initial_guess=P_prev)
        
        if rate > 1e-18:
            P_prev = params.copy()
            
        results.append({
            "fiber_length": float(L),
            "n_X": int(n_X),
            "key_rate": float(rate),
            "optimized_parameters": {
                "mu_1": float(params[0]), "mu_2": float(params[1]), 
                "P_mu_1": float(params[2]), "P_mu_2": float(params[3]), 
                "P_X_value": float(params[4])
            }
        })
        
    duration = time.time() - start_time
    logging.info(f"✅ Finished n_X={n_X:.0e} in {duration:.2f}s")
    return n_X, results, duration

def run_jax_pipeline():
    # Setup
    base_dir = os.path.join(os.path.dirname(script_dir), "outputs", "test_tier3_short")
    os.makedirs(base_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Constants
    bounds = [(4e-4, 0.9), (2e-4, 0.5), (1e-6, 1.0-1e-6), (1e-6, 1.0-1e-6), (1e-6, 1.0-1e-6)]
    
    # MODIFIED: Reduced L points and only one n_X for testing
    L_values = np.linspace(0, 200, 20) # 20 points for quick test
    n_X_values = [1e4]
    
    constants = (0.2, 0.1, 6e-7, 1e-10, 1e-15, 1.16, 0.01, 0, 1) # Matches physics signature

    final_results = []
    timing_per_nx = {}
    total_time = 0.0
    start_wall = time.time()
    
    logging.info("Starting optimization in TEST mode (n_X=1e4, 20 L points)...")
    
    # Run sequentially
    for n_X in n_X_values:
        try:
            n_X_res, res, dur = run_single_nx_curve(n_X, L_values, bounds, constants)
            final_results.extend(res)
            timing_per_nx[str(float(n_X))] = dur
            total_time += dur
            logging.info(f"Completed n_X={n_X} in {dur:.2f}s")
        except Exception as e:
            logging.error(f"Failed to process n_X={n_X}: {e}")
            import traceback
            logging.error(traceback.format_exc())

    end_wall = time.time()

    # Save
    json_path = os.path.join(run_dir, f"reordered_qkd_results_jax_{timestamp}.json")
    grouped = {}
    for r in final_results:
        k = str(float(r['n_X']))
        if k not in grouped: grouped[k] = []
        grouped[k].append(r)
    
    with open(json_path, 'w') as f: json.dump(grouped, f, indent=2)
    
    # Report
    wall_duration = time.time() - start_wall
    try:
        generate_timing_report.generate_report(timing_per_nx, total_time, wall_duration, os.cpu_count(), run_dir, "TestGlobalJAX_")
        plot_final_results.generate_plots(json_path, "TestGlobalJAX_")
        print(f"Plot saved to: {run_dir}")
    except Exception as e:
        print(f"Reporting error: {e}")

if __name__ == "__main__":
    run_jax_pipeline()
