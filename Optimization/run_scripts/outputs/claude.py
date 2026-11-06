import os
# --- PREVENTION GUARDS ---
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
from scipy.optimize import differential_evolution, minimize
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

# Initial guesses per n_X (from EvolutionDifferentiation)
INITIAL_GUESSES_MAP = {
    1e4: np.array([0.65, 0.15, 0.05, 0.61, 0.425]),
    1e5: np.array([0.62, 0.24, 0.10, 0.70, 0.55]),
    1e6: np.array([0.68, 0.30, 0.14, 0.74, 0.66]),
    1e7: np.array([0.55, 0.34, 0.15, 0.75, 0.75]),
    1e8: np.array([0.54, 0.375, 0.16, 0.775, 0.83]),
    1e9: np.array([0.52, 0.40, 0.18, 0.785, 0.88]),
}
DEFAULT_GUESS = np.array([0.52, 0.40, 0.18, 0.785, 0.88])

# JIT-compiled objective and gradient
objective_val_and_grad = jit(value_and_grad(scalar_objective, argnums=0))


def optimize_exactly_like_evolution_differentiation(L, n_X, bounds, constants, P_prev):
    """
    EXACT COPY of Evolution Differentiation's optimize_batch_jax function.
    
    This is the PROVEN approach that gives smooth, accurate results.
    No modifications, no optimizations - just copy what works.
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
    candidates.append(P_prev.copy())
        
    # Candidate 2: Small Jitter (Local Exploration) - 5% jitter
    jitter_scale = 0.05 
    jittered = P_prev * np.random.uniform(1.0 - jitter_scale, 1.0 + jitter_scale, size=len(P_prev))
    jittered = np.clip(jittered, [b[0] for b in bounds_scipy], [b[1] for b in bounds_scipy])
    candidates.append(jittered)
    
    # Candidate 3: Global Reset (Safety Net)
    global_reset_guess = INITIAL_GUESSES_MAP.get(n_X, DEFAULT_GUESS)
    candidates.append(global_reset_guess.copy())
            
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
                options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 1500}
            )
            rate = -res.fun
            if rate > best_rate and res.success:
                best_rate = rate
                best_params = res.x
        except Exception:
            continue

    # --- STEP 3: GLOBAL ESCAPE MECHANISM ---
    # EXACT copy from Evolution Differentiation - uses differential_evolution
    if best_rate < 1e-12:  # Threshold for "Dead Basin"
        try:
            # Deterministic Seed based on L
            L_seed = int(L * 100) + 42
            
            # Use differential_evolution EXACTLY as Evolution Differentiation does
            result_global = differential_evolution(
                func=scipy_fun_val_only,
                bounds=bounds_scipy,
                maxiter=100,       # Modest global search
                popsize=20,       
                polish=False,      # We'll do our own polishing
                disp=False,
                seed=L_seed,
                workers=1 
            )
            
            # Refine Global Result with strict L-BFGS-B
            res_refined = minimize(
                fun=scipy_fun,
                x0=result_global.x,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds_scipy,
                options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 1500}
            )
            
            final_global_rate = -res_refined.fun
            
            # Only switch if Global found something significantly better
            if final_global_rate > best_rate and final_global_rate > 1e-12:
                best_rate = final_global_rate
                best_params = res_refined.x
                
        except Exception as e:
            pass  # Global search failed, accept local fate

    # Final Sanity Check
    if best_params is None: 
        best_params = global_reset_guess
        best_rate = 0.0
    
    # If essentially 0, clamp to 0 for clean plotting
    if best_rate < 1e-15:
        best_rate = 0.0

    return best_rate, best_params


def run_single_nx_curve(n_X, L_values, bounds, constants):
    """Process a single n_X curve"""
    process_id = os.getpid()
    logging.info(f"Worker {process_id} starting n_X={n_X:.0e} curve (Evolution Differentiation approach)")
    
    start_time = time.time()
    P_prev = INITIAL_GUESSES_MAP.get(n_X, DEFAULT_GUESS)
    results = []
    
    for i, L in enumerate(L_values):
        rate, params = optimize_exactly_like_evolution_differentiation(
            L, n_X, bounds, constants, P_prev
        )
        
        # Update P_prev for next L-point (continuity)
        if rate > 1e-20 and not np.isnan(params).any():
            P_prev = params.copy()
        
        results.append({
            "fiber_length": float(L), 
            "n_X": int(n_X), 
            "key_rate": float(rate),
            "optimized_parameters": {
                "mu_1": float(params[0]), 
                "mu_2": float(params[1]),
                "P_mu_1": float(params[2]), 
                "P_mu_2": float(params[3]),
                "P_X_value": float(params[4])
            }
        })
        
        # Progress logging
        if i % 100 == 0:
            logging.info(f"Worker {process_id} | n_X={n_X:.0e} | L={L:.1f} | Rate={rate:.2e}")
    
    duration = time.time() - start_time
    logging.info(f"✅ Worker {process_id} finished n_X={n_X:.0e} in {duration:.2f}s")
    
    return n_X, results, duration


def run_pipeline():
    """Main pipeline with optimized parallelization"""
    
    # Setup output directory
    base_dir = os.path.join(os.path.dirname(script_dir), "outputs", "tier3_evolution_exact_copy")
    os.makedirs(base_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Problem setup (EXACT same as EvolutionDifferentiation)
    bounds = [(4e-4, 0.9), (2e-4, 0.5), (1e-6, 1.0-1e-6), (1e-6, 1.0-1e-6), (1e-6, 1.0-1e-6)]
    L_values = np.linspace(0, 200, 1000)  # Same 1000 points
    n_X_values = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
    constants = (0.2, 0.1, 6e-7, 1e-10, 1e-15, 1.16, 0.01, 0, 1)
    
    final_results = []
    timing_per_nx = {}
    total_time_sequential = 0.0
    start_wall = time.time()
    
    # KEY OPTIMIZATION: Use more cores since Evolution Differentiation is sequential
    # Each n_X curve is independent, so we can run all 6 in parallel
    max_workers = 6
    logging.info(f"🚀 Starting Evolution Differentiation (Exact Copy) Optimization...")
    logging.info(f"   Algorithm: Same as best-performing script")
    logging.info(f"   Workers: {max_workers} (parallel over n_X)")
    logging.info(f"   Output: {run_dir}")
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_single_nx_curve, n_X, L_values, bounds, constants): n_X 
            for n_X in n_X_values
        }
        
        for future in concurrent.futures.as_completed(futures):
            n_X_done = futures[future]
            try:
                nx_key, res, dur = future.result()
                final_results.extend(res)
                timing_per_nx[str(float(nx_key))] = dur
                total_time_sequential += dur
                logging.info(f"✅ Completed n_X={nx_key:.0e}: {len(res)} points in {dur:.2f}s")
            except Exception as e:
                logging.error(f"❌ n_X={n_X_done:.0e} failed: {e}")
                import traceback
                logging.error(traceback.format_exc())
    
    # Save results
    json_path = os.path.join(run_dir, f"evolution_exact_copy_{timestamp}.json")
    grouped = {}
    for r in final_results:
        k = str(float(r['n_X']))
        if k not in grouped:
            grouped[k] = []
        grouped[k].append(r)
    
    with open(json_path, 'w') as f:
        json.dump(grouped, f, indent=2)
    
    # Generate reports
    wall_duration = time.time() - start_wall
    
    try:
        generate_timing_report.generate_report(
            timing_per_nx, total_time_sequential, wall_duration, max_workers, run_dir, "EvolutionExact_"
        )
        logging.info("✅ Timing report generated")
    except Exception as e:
        logging.error(f"⚠️ Timing report error: {e}")
    
    try:
        plot_final_results.generate_plots(json_path, "EvolutionExact_")
        logging.info("✅ Plots generated")
    except Exception as e:
        logging.error(f"⚠️ Plot generation error: {e}")
    
    # Summary
    logging.info(f"")
    logging.info(f"{'='*60}")
    logging.info(f"OPTIMIZATION COMPLETE - EVOLUTION DIFFERENTIATION (EXACT COPY)")
    logging.info(f"{'='*60}")
    logging.info(f"Total points computed: {len(final_results)}")
    logging.info(f"Sequential time: {total_time_sequential:.2f}s ({total_time_sequential/60:.2f} min)")
    logging.info(f"Wall clock time: {wall_duration:.2f}s ({wall_duration/60:.2f} min)")
    logging.info(f"Speedup: {total_time_sequential/wall_duration:.2f}x")
    logging.info(f"Results saved to: {json_path}")
    logging.info(f"{'='*60}")
    logging.info(f"")
    logging.info(f"This should match Evolution Differentiation's accuracy exactly.")
    logging.info(f"Speedup from parallelization: ~{max_workers}x faster wall time")


if __name__ == "__main__":
    run_pipeline()