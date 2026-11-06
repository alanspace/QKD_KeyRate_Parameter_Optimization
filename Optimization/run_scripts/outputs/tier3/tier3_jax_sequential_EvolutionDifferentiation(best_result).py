import os
import sys
import time
import json
import functools
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, value_and_grad
import concurrent.futures
import generate_timing_report
import plot_final_results

# Ensure the script uses the correct project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import physics logic (Assume these are JAX-compatible)
from src.qkd.model import calculate_key_rates_and_metrics, objective_val_and_grad, objective
# Note: typically 'objective' is the scalar function, we will wrap it for JAX if needed.
# But 'objective_val_and_grad' likely returns (val, grad). 
# For JAX vmap to work best, we should use a pure JAX function 'objective_jax'.
# Since I cannot see src/qkd/model.py internals deeply, I will assume 'objective' can be compiled
# or I will reconstruct the loop.
# Actually, 'objective_val_and_grad' is likely manually differentiating or using jax.grad.
# Let's import the raw 'objective' and use jax.value_and_grad on it to be safe and purely functional.

from src.qkd.model import objective, scalar_objective 

# --- CONFIGURATION ---
BATCH_SIZE = 5000  # Number of parallel starting points per L. Massive parallelism!
LEARNING_RATE = 0.01
ITERATIONS = 300
ENABLE_FLOAT64 = True # CPU supports float64, so let's use it for accuracy.

if ENABLE_FLOAT64:
    jax.config.update("jax_enable_x64", True)
else:
    jax.config.update("jax_enable_x64", False)

# FORCE CPU due to Metal/MPS 'default_memory_space' unimplemented error
# This ensures robust execution while still benefiting from AVX/SIMD parallelism.
jax.config.update("jax_platform_name", "cpu")

# Try to use Metal (GPU)
try:
    print(f"🚀 JAX Devices available: {jax.devices()}")
except:
    print("⚠️ JAX could not find devices, falling back to default.")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Wrapper to return SCALAR key rate for differentiation
def objective_scalar(params, L, n_X, *constants):
    # objective returns a tuple (key_rate, metrics...)
    # We strip it to just return the key rate (scalar)
    res = objective(params, L, n_X, *constants)
    
    # Debug: Check if key rate is NaN
    # jax.debug.print("Rate: {r}", r=res[0]) 
    return res[0] # Return just the penalized key rate

@jit
def update_step(params, L, n_X, constants, bounds_min, bounds_max):
    """
    Performs one step of Projected Gradient Descent on a batch of parameters.
    """
    (val, grads) = value_and_grad(objective_scalar, argnums=0)(
        params, L, n_X, *constants
    )
    
    # Debug NaNs in Gradients
    # We use jax.debug.print to see if grads are NaN for the FIRST element of batch usually,
    # but here 'params' is a single instance (vmapped outside).
    # So we print if ANY nan.
    
    # Monitor gradient norms
    # jax.debug.print("Grads: {g}", g=grads)
    # jax.debug.print("Val: {v}", v=val)
    
    # Update (Gradient Descent on Negative Rate)
    # Gradient Descent: p_new = p - lr * grad
    new_params = params - LEARNING_RATE * grads
    
    # Clip to Box Bounds first
    new_params = jnp.clip(new_params, bounds_min, bounds_max)
    
    # --- PROJECTION STEP (Projected Gradient) ---
    mu_1 = new_params[0]
    mu_2 = new_params[1]
    P_mu_1 = new_params[2]
    P_mu_2 = new_params[3]
    P_X = new_params[4]
    
    mu_3 = 2e-4
    epsilon = 1e-4 # margins
    
    # Constraint 1: P_mu_1 + P_mu_2 <= 0.999
    # If violation, scale them down
    sum_P = P_mu_1 + P_mu_2
    scale_factor = jnp.where(sum_P > 0.999, 0.999 / (sum_P + 1e-12), 1.0)
    P_mu_1 *= scale_factor
    P_mu_2 *= scale_factor
    
    # Constraint 2: mu_2 >= mu_3 + epsilon
    mu_2 = jnp.maximum(mu_2, mu_3 + epsilon)
    
    # Constraint 3: mu_1 >= mu_2 + mu_3 + epsilon
    mu_1 = jnp.maximum(mu_1, mu_2 + mu_3 + epsilon)
    
    # Re-pack
    new_params = jnp.array([mu_1, mu_2, P_mu_1, P_mu_2, P_X])
    
    # Final clip
    new_params = jnp.clip(new_params, bounds_min, bounds_max)
    
    return new_params, val

from scipy.optimize import differential_evolution, minimize

# --- CONFIG: Smoothness-First Optimization ---
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

def optimize_batch_jax(L, n_X, bounds, constants, initial_guess=None):
    """
    Smoothness-First Optimization (Global n Local Warm Start).
    
    Strategy:
    1. Local Bias: Race 'Warm Start' (continuity) and 'Small Jitter' (local exploration).
       - Keeps parameters evolving smoothly along the basin.
    2. Global Reset: Include the 'Safe Guess' to recover if warm start drifts too far.
    3. Global Escape: Only run heavy Global Search if local candidates fail (rate -> 0).
       - Provides a safety net for physical cliffs without adding noise to smooth regions.
    """
    
    # 1. Wrapper for JAX Objective
    @jit
    def jax_obj_and_grad(p):
        v, g = value_and_grad(scalar_objective, argnums=0)(p, L, n_X, *constants)
        return v, g

    def scipy_fun(p):
        p_jax = jnp.array(p)
        val, grad = jax_obj_and_grad(p_jax)
        return float(val), np.array(grad, dtype=np.float64)
        
    def scipy_fun_val_only(p):
        p_jax = jnp.array(p)
        return float(scalar_objective(p_jax, L, n_X, *constants))

    bounds_scipy = bounds
    
    # --- STEP 1: DEFINE SMOOTH CANDIDATES ---
    candidates = []
    
    # Candidate 1: Warm Start (The "Smoothness Constraint")
    # Must be first to win tie-breakers if rates are identical
    if initial_guess is not None:
        candidates.append(initial_guess)
        
    # Candidate 2: Small Jitter (Local Exploration)
    # 5% jitter to help nudge out of saddle points without jumping basins
    if initial_guess is not None:
        jitter_scale = 0.05 
        candidates.append(initial_guess * np.random.uniform(1.0 - jitter_scale, 1.0 + jitter_scale, size=len(initial_guess)))
    
    # Candidate 3: Global Reset (Safety Net)
    # The "Architect's" value for this n_X
    global_reset_guess = INITIAL_GUESSES_MAP.get(n_X, DEFAULT_GUESS)
    candidates.append(global_reset_guess)
            
    best_rate = -1.0
    best_params = None
    
    # --- STEP 2: RACE CANDIDATES (Local Search) ---
    for start_point in candidates:
        start_point = np.clip(start_point, [b[0] for b in bounds_scipy], [b[1] for b in bounds_scipy])
        try:
            res = minimize(
                fun=scipy_fun,
                x0=start_point,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds_scipy,
                options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 1500} # Strict Precision
            )
            rate = -res.fun
            if rate > best_rate and res.success:
                best_rate = rate
                best_params = res.x
        except Exception:
            continue

    # --- STEP 3: GLOBAL ESCAPE MECHANISM ---
    # Only trigger if local search Failed (Rate is effectively zero)
    # This prevents "Stochastic Roughness" in valid regions, but handles cliffs.
    if best_rate < 1e-12: # Threshold for "Dead Basin"
        try:
            # Deterministic Seed based on L (to minimize drift if global is called)
            L_seed = int(L * 100) + 42
            
            result_global = differential_evolution(
                func=scipy_fun_val_only,
                bounds=bounds_scipy,
                maxiter=100,      # Modest global search
                popsize=20,       
                polish=False,       
                disp=False,
                seed=L_seed,
                workers=1 
            )
            
            # Refine Global Result
            res_refined = minimize(
                fun=scipy_fun,
                x0=result_global.x,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds_scipy,
                options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 1500}
            )
            
            final_global_rate = -res_refined.fun
            
            # Only switch if Global found something significantly better (alive vs dead)
            if final_global_rate > best_rate and final_global_rate > 1e-12:
                best_rate = final_global_rate
                best_params = res_refined.x
                
        except Exception as e:
            pass # Global search failed, accept local fate

    # Final Sanity Check
    if best_params is None: 
         best_params = global_reset_guess
         best_rate = 0.0
    
    # If essentially 0, clamp to 0 for clean plotting
    if best_rate < 1e-15:
        best_rate = 0.0

    return best_rate, best_params

# Argument Unpacking Helper
# We need to pass constants as a tuple to allow JAX to hash/pass them.
# constants = (alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event)

def find_global_max_rate(L, n_X, bounds, constants):
    """
    Step 1 of Danger Zone: Find absolute maximum key rate R_max(L).
    Uses heavy DE similar to run_global_refined.
    """
    @jit
    def jax_obj_and_grad(p):
        v, g = value_and_grad(scalar_objective, argnums=0)(p, L, n_X, *constants)
        return v, g

    def scipy_fun(p):
        p_jax = jnp.array(p)
        val, grad = jax_obj_and_grad(p_jax)
        return float(val), np.array(grad, dtype=np.float64)
        
    def scipy_fun_val_only(p):
        p_jax = jnp.array(p)
        return float(scalar_objective(p_jax, L, n_X, *constants))

    bounds_scipy = bounds
    
    # Heavy Global Search
    L_seed = int(L * 100) + 42
    result_global = differential_evolution(
        func=scipy_fun_val_only,
        bounds=bounds_scipy,
        maxiter=150,     
        popsize=20,       
        polish=False,       
        disp=False,
        seed=L_seed,
        workers=1 
    )
    
    # Strict local refinement
    try:
        res_refined = minimize(
            fun=scipy_fun,
            x0=result_global.x,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds_scipy,
            options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 1500}
        )
        R_max = -res_refined.fun
        P_true = res_refined.x
    except Exception:
        R_max = -result_global.fun if hasattr(result_global, 'fun') else 0.0
        P_true = result_global.x
    
    if R_max < 1e-15: R_max = 0.0
    return R_max, P_true

def find_smooth_params(L, n_X, bounds, constants, R_max, P_start):
    """
    Step 2 of Danger Zone: Find smoothest parameter set P_smooth.
    Minimizes ||P - P_start||² subject to R(P) >= R_max - epsilon.
    """
    epsilon = 1e-14
    M = 1e12
    
    P_start_jax = jnp.array(P_start)
    R_max_jax = jnp.array(R_max)
    
    @jit
    def smoothness_objective(P):
        neg_rate = scalar_objective(P, L, n_X, *constants)
        R_current = -neg_rate
        l2_penalty = jnp.sum((P - P_start_jax)**2)
        rate_deficit = (R_max_jax - epsilon) - R_current
        constraint_penalty = M * jnp.maximum(0.0, rate_deficit)
        return l2_penalty + constraint_penalty
    
    smoothness_val_and_grad = jit(value_and_grad(smoothness_objective, argnums=0))
    
    def scipy_smooth_fun(p):
        p_jax = jnp.array(p)
        val, grad = smoothness_val_and_grad(p_jax)
        return float(val), np.array(grad, dtype=np.float64)
    
    try:
        res_smooth = minimize(
            fun=scipy_smooth_fun,
            x0=P_start,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 1500}
        )
        P_smooth = res_smooth.x
        # Verify
        R_actual = -float(scalar_objective(jnp.array(P_smooth), L, n_X, *constants))
        if R_actual < R_max - epsilon - 1e-10:
            return R_max, P_start # Fallback
    except:
        return R_max, P_start

    return -scalar_objective(jnp.array(P_smooth), L, n_X, *constants), P_smooth

def optimize_adaptive(L, n_X, bounds, constants, initial_guess=None):
    """
    Intelligent Adaptive Strategy:
    - Uses local 'optimize_batch_jax' in the Stable Zone.
    - Uses heavy 'Two-Step Global-Constrained' in the Danger Zone.
    """
    # Determine Danger Zone Threshold for this n_X
    danger_start = DANGER_ZONE_STARTS.get(n_X, 150.0) # Default to 150 if not found
    
    if L < danger_start:
        # --- STABLE ZONE ---
        return optimize_batch_jax(L, n_X, bounds, constants, initial_guess)
    else:
        # --- DANGER ZONE ---
        # Step 1: Global Maximum
        R_max, P_true = find_global_max_rate(L, n_X, bounds, constants)
        
        # Step 2: Smoothness Constraint
        # We need a previous smooth point for this to work effectively. 
        # If initial_guess is None, we just return P_true
        if initial_guess is None:
            return R_max, P_true
            
        R_smooth, P_smooth = find_smooth_params(L, n_X, bounds, constants, R_max, initial_guess)
        
        # Step 3: Best-Effort Decision
        # If smooth search failed to satisfy the constraint (due to physical discontinuity or local minima),
        # we must jump to the global maximum.
        if R_smooth >= R_max - 1e-12:
            return R_smooth, P_smooth
        else:
            return R_max, P_true

# --- SPLIT CONFIGURATION (New Global Variable) ---
# Maps n_X to the number of segments to split the L-array into.
# SEQUENTIAL VERSION: No splitting
SPLIT_SEGMENTS_MAP = {
    1e4: 1, 
    1e5: 1,  
    1e6: 1,  
    1e7: 1,
    1e8: 1,
    1e9: 1
}

def _run_sequential_segment(n_X, L_segment, bounds, constants, P_start_for_segment, log_prefix):
    """
    Runs a single, continuous, sequential L-loop over a segment of L-values.
    """
    start_time = time.time()
    results_for_segment = []
    
    # Initialize the continuity chain with the provided cold start
    P_smooth_prev = P_start_for_segment.copy()

    total_points = len(L_segment)
    for i, L in enumerate(L_segment):
        if i % (max(1, total_points // 10)) == 0:
             print(f"   ... {log_prefix} processing L={L:.1f}km ({i}/{total_points})")

        # --- Adaptive Optimization Call (The Core Logic) ---
        rate, params = optimize_adaptive(L, n_X, bounds, constants, initial_guess=P_smooth_prev)
        
        # Update P_smooth_prev for the next L-point
        # Only update if valid result to maintain continuity
        if rate > 1e-20 and not np.isnan(params).any():
            P_smooth_prev = params.copy()
            
        # Store result
        result_dict = {
            "fiber_length": float(L),
            "n_X": int(n_X),
            "key_rate": float(rate),
            "optimized_parameters": {
                "mu_1": float(params[0]), "mu_2": float(params[1]), "P_mu_1": float(params[2]), 
                "P_mu_2": float(params[3]), "P_X_value": float(params[4])
            },
            "initial_guess": {}
        }
        results_for_segment.append(result_dict)
        
    duration = time.time() - start_time
    # This is the worker's success report
    logging.info(f"✅ Segment {log_prefix} completed in {duration:.2f}s") 
    
    return results_for_segment, duration

def run_single_nx_curve(n_X, L_values, bounds, constants):
    start_time_total = time.time()
    
    # SEQUENTIAL VERSION: No splitting logic needed, just run directly
    initial_P_start = INITIAL_GUESSES_MAP.get(n_X, DEFAULT_GUESS)
    
    results, duration = _run_sequential_segment(
        n_X, L_values, bounds, constants, initial_P_start, f"n_X={n_X:.0e}-SEQ"
    )

    return n_X, results, duration

def run_jax_pipeline():
    # Pre-calculate Run Directory
    script_dir_out = os.path.dirname(os.path.abspath(__file__))
    base_output_dir = os.path.join(script_dir_out, "results_global_jax")
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Setup Logging
    log_file = os.path.join(run_dir, "optimization_jax.log")
    # Reset logger to avoid duplicate handlers if run multiple times
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logging.info(f"🚀 Starting JAX Optimization. Logs at {log_file}")
    logging.info(f"📂 Run Directory: {run_dir}")

    # Constants
    bounds = [(4e-4, 0.9), (2e-4, 0.5), (1e-6, 1.0 - 1e-6), (1e-6, 1.0 - 1e-6), (1e-6, 1.0 - 1e-6)] # Relaxed bounds for float32
    # FAST RUN FOR VERIFICATION
    # L_values = np.linspace(0, 200, 5) 
    # n_X_values = [1e4]
    # "Official" High-Resolution Run Settings (0.2 km steps)
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

    final_results = []
    
    print(f"🏎️  Starting JAX Optimization (Parallel Scipy Minimization)")
    print(f"📂 Output Folder: {run_dir}")
    
    filename = f"qkd_results_jax_{timestamp}.json"
    filepath = os.path.join(run_dir, "reordered_" + filename) 
    
    grouped_data = {}

    # --- CRITICAL FIX: Specific Initial Guesses for each n_X ---
    # From the Architect's Notebook logic
    initial_guesses_map = {
        1e4: np.array([0.65, 0.15, 0.05, 0.61, 0.425]),
        1e5: np.array([0.62, 0.24, 0.10, 0.70, 0.55]),
        1e6: np.array([0.68, 0.30, 0.14, 0.74, 0.66]),
        1e7: np.array([0.55, 0.34, 0.15, 0.75, 0.75]),
        1e8: np.array([0.54, 0.375, 0.16, 0.775, 0.83]),
        1e9: np.array([0.52, 0.40, 0.18, 0.785, 0.88]),
    }
    default_guess = np.array([0.52, 0.40, 0.18, 0.785, 0.88])

    
    # --- PARALLEL EXECUTION OVER n_X VALUES ---
    MAX_WORKERS = os.cpu_count()
    print(f"🏎️  Starting Parallel Optimization using {MAX_WORKERS} cores...")
    logging.info(f"Using {MAX_WORKERS} parallel workers")
    
    final_results = []
    total_time = 0.0
    timing_per_nx = {}
    
    start_time_wall_clock = time.time()
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all n_X jobs to the pool
        future_to_nx = {
            executor.submit(run_single_nx_curve, n_X, L_values, bounds, constants): n_X 
            for n_X in n_X_values
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_nx):
            n_X_done = future_to_nx[future]
            try:
                n_X_result, batch_results, duration = future.result()
                final_results.extend(batch_results)
                total_time += duration
                
                # Store in grouped_data
                nx_str = str(float(n_X_result))
                grouped_data[nx_str] = batch_results
                
                # Store timing
                timing_per_nx[nx_str] = duration
                
                msg = f"✅ Finished n_X={n_X_result:.0e} in {duration:.2f}s ({len(batch_results)} points)"
                print(msg)
                logging.info(msg)
                
                # Save and plot progressively
                with open(filepath, 'w') as f:
                    json.dump(grouped_data, f, indent=2)
                # plot_jax_results is deprecated

                
            except Exception as exc:
                msg = f"❌ n_X={n_X_done:.0e} generated an exception: {exc}"
                print(msg)
                logging.error(msg)
    
    # Log total sequential time
    msg = f"🎯 Total optimization time (sum of all n_X): {total_time:.2f}s ({total_time/60:.2f} min)"
    print(msg)
    logging.info(msg)
                
    # Save Results
    # (filepath is already defined above using run_dir)
    
    # Verify we actually have meaningful results
    if not final_results:
        print("❌ Fatal: No results generated! Check optimizer.")
        return

    # print sample
    print(f"Sample Result: {final_results[0]}")

    grouped_data = {}
    for entry in final_results:
        nx = str(float(entry["n_X"]))
        if nx not in grouped_data: grouped_data[nx] = []
        grouped_data[nx].append(entry)
        
    # Save Grouped Data
    with open(filepath, 'w') as f:
        json.dump(grouped_data, f, indent=2)
        
    print(f"✅ Saved JAX results to {filepath} with {len(final_results)} entries.")
    
    # --- PLOT & REPORT GENERATION ---
    print("\n📊 Generating Plots and Reports...")
    
    # 1. Generate Timing Report
    wall_duration = time.time() - start_time_wall_clock
    
    try:
        generate_timing_report.generate_report(
            timing_data=timing_per_nx,
            total_sequential=total_time,
            wall_time=wall_duration,
            num_cores=MAX_WORKERS,
            output_dir=run_dir 
        )
    except Exception as e:
        print(f"⚠️ Error generating timing report: {e}")
        
    # 2. Generate Summary & Detailed Plots (Using plot_final_results logic)
    # Call generate_plots with the specific JSON file for this run
    try:
        plot_final_results.generate_plots(filepath)
    except Exception as e:
        print(f"⚠️ Error generating final plots: {e}")

    # 3. Generate Simple JAX Plots
    # Deprecated: plot_jax_results was removed in favor of plot_final_results
    pass


if __name__ == "__main__":
    run_jax_pipeline()
