import os
import sys
import time
import json
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.optimize import minimize, dual_annealing
import concurrent.futures
import jax

# Add project root to path
sys.path.append(os.getcwd())

# JAX Configuration (MUST be before other JAX usage)
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

# Import from your project
from src.qkd.model import objective_val_and_grad

# Setup basic logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variables for initial guess (similar to notebook)
x0 = np.array([0.52, 0.40, 0.18, 0.785, 0.88]) 

def optimize_single_instance(L, n_X, bounds, initial_guess, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event):
    """Optimize key rate for a given fiber length L and fixed n_X using JAX Gradients."""
    # Note: globals in multiprocessing can be tricky, using local best logic
    current_x0 = initial_guess.copy()

    try:
        # Debug initial value
        init_val, init_grad = objective_val_and_grad(
            current_x0, L, n_X, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event
        )
        if L == 0.0: # Print check for first item
             print(f"DEBUG L=0 Init: Val={init_val}, KeyRate={-init_val}, GradNorm={np.linalg.norm(init_grad)}")

        # Wrapper that returns (value, gradient) for Scipy
        def objective_wrapper(params):
            val, grad = objective_val_and_grad(
                params, L, n_X, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event
            )
            # Clip gradient to prevent explosions
            grad = np.clip(np.array(grad), -1e6, 1e6)
            return float(val), np.array(grad)

        # Global optimization (Annealing) uses VALUE ONLY
        # We pass 'jac' in minimizer_kwargs for the local search phase of annealing
        global_result = dual_annealing(
            func=lambda p: objective_wrapper(p)[0], 
            bounds=bounds,
            x0=current_x0,
            maxiter=50, # Reduced iterations due to better local search
            minimizer_kwargs={
                'method': 'L-BFGS-B', 
                'jac': lambda p: objective_wrapper(p)[1],
                'options': {'ftol': 1e-8}
            }
        )

        # Final Local Refinement with L-BFGS-B (Gradient-Based)
        local_result = minimize(
            fun=objective_wrapper, 
            x0=global_result.x,
            method='L-BFGS-B',
            jac=True, # Function returns (val, grad)
            bounds=bounds,
            options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-12}
        )

        optimized_params = local_result.x
        optimized_key_rate = -local_result.fun # Invert back to positive key rate

        # Simple verification
        if optimized_key_rate < 0: optimized_key_rate = 0.0

        return L, n_X, optimized_key_rate, optimized_params, initial_guess

    except Exception as e:
        print(f"❌ Error at L={L}: {e}")
        return L, n_X, 0.0, initial_guess, initial_guess

def main():
    print(f"🚀 Starting Compressed Optimization Run on {jax.devices()[0]}")
    
    # Parameters
    bounds = [(4e-4, 0.9), (2e-4, 0.5), (1e-12, 1.0 - 1e-12), (1e-12, 1.0 - 1e-12), (1e-12, 1.0 - 1e-12)]
    
    # Run a subset for demonstration (0 to 50km is usually the most interesting part)
    # Full run: np.linspace(0, 200, 1000)
    L_values = np.linspace(0, 50, 50) 
    n_X_values = [1e9] 
    
    alpha = 0.2
    eta_Bob = 0.1
    P_dc_value = 6e-7
    epsilon_sec = 1e-10
    epsilon_cor = 1e-15
    f_EC = 1.16
    e_mis = 0.01
    P_ap = 0
    n_event = 1

    start_time = time.time()

    # Parallel Execution
    # Using ProcessPoolExecutor for CPU-bound (or ThreadPool if JAX releases GIL nicely, JAX-Metal is usually fine with Threads)
    # Since we are using JAX Metal, ThreadPoolExecutor is preferred to avoid multiprocessing overhead and re-compiling JAX kernels.
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(optimize_single_instance, L, n_X, bounds, x0, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event)
            for L in L_values
            for n_X in n_X_values
        ]
        
        results = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Optimizing"):
             results.append(future.result())

    end_time = time.time()
    duration = end_time - start_time
    
    # Sort results
    results.sort(key=lambda x: x[0]) # Sort by L

    print(f"\n✅ Optimization Complete in {duration:.2f} seconds!")
    print(f"Speed: {len(results)/duration:.2f} it/s")
    
    # Show first few results
    print("\nSample Results:")
    for res in results[:5]:
        print(f"L={res[0]:.1f}km -> KeyRate={res[2]:.2e}")

    # Save small dataset
    dataset = [{
        "fiber_length": float(r[0]),
        "n_X": int(r[1]),
        "key_rate": float(r[2]),
        "optimized_params": list(r[3])
    } for r in results]
    
    with open('optimized_results_fast.json', 'w') as f:
        json.dump(dataset, f, indent=2)

if __name__ == "__main__":
    main()
