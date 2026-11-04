import os
import sys
import time
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import glob
from scipy.optimize import minimize

# --- PATH SETUP ---
# Determine absolute paths based on the script location
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path) # Optimization/run_scripts/JAX_variants
optimization_dir = os.path.abspath(os.path.join(script_dir, "../../")) # Optimization
project_root = os.path.dirname(optimization_dir) # Project Root
tools_dir = os.path.join(optimization_dir, "tools")

# Add paths to sys.path
if project_root not in sys.path:
    sys.path.append(project_root)
if tools_dir not in sys.path:
    sys.path.append(tools_dir)

# Import Physics & Tools
from src.qkd.model import objective_val_and_grad, scalar_objective
import generate_timing_report

# --- CONFIGURATION ---
# JAX configuration
import jax
from jax import jacfwd, jacrev
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu") # Force CPU for 64-bit precision

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- JIT COMPILE FORWARD-OVER-REVERSE HESSIAN ---
# We replace jax.hessian (Reverse-over-Reverse) with jacfwd(jacrev) (Forward-over-Reverse).
# This is more numerically stable for small gradients (long distance tail) and 
# compiles faster on M2 unified memory architecture.
hessian_func = jax.jit(jacfwd(jacrev(scalar_objective)))

def optimize_single_nx_sequence_for(n_X, L_values, bounds, initial_guess, constants):
    """
    Optimizes key rate using Bi-Directional Homotopy (Forward + Reverse Sweeps).
    1. Forward Pass (0 -> 200 km): Standard optimization.
    2. Reverse Pass (200 -> 0 km): Uses stable tail parameters to backfill blind spots.
    3. Merge: Picks the best result for each distance L.
    """
    alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event = constants
    
    # constants
    lambda_reg = 1e-6 # Tikhonov Regularization

    # helper for minimization
    def objective_wrapper(params, L_val, prev_params=None):
        val, grad = objective_val_and_grad(
            params, L_val, n_X, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event
        )
        
        # Add "Stability" penalty (Tikhonov)
        if prev_params is not None:
            reg_loss = lambda_reg * np.sum((params - prev_params)**2)
            reg_grad = 2 * lambda_reg * (params - prev_params)
            val += reg_loss
            grad += reg_grad

        if not np.isfinite(val): return 1e9, np.zeros_like(params)
        return float(val), np.array(grad)

    def hessian_wrapper(params, L_val, prev_params=None):
        H = hessian_func(
            jnp.array(params), L_val, n_X, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event
        )
        H = np.array(H)
        
        # Add Regularization Hessian (2 * lambda * I)
        if prev_params is not None:
             H += 2 * lambda_reg * np.eye(len(params))
             
        return H

    # --- SHARED OPTIMIZER CORE ---
    def run_sweep(L_seq, start_guess):
        sweep_results = {}
        current_guess = start_guess.copy()
        prev_params = None # No regularization for the very first point of the sweep sequence
        
        for L in L_seq:
            try:
                # Stage 0: Hybrid Warming (Scout) - Optional, kept for robustness
                # We skip regularization here to let it find the basin freely? 
                # Or include it? User's plan implies Hessian is the main stage. 
                # Let's keep Scout simple or verify. 
                # User's plan replaces the loop structure. 
                # I will keep "Scout" but disable regularization for it, or just rely on Stage 1.
                # Actually, let's include reg in Scout if available to avoid jumping.
                
                # STAGE 0: Hybrid Warming (Scout)
                # Helps finding the correct basin for the Hessian.
                res_warm = minimize(
                    fun=lambda p: objective_wrapper(p, L, prev_params),
                    x0=current_guess,
                    method='L-BFGS-B',
                    jac=True,
                    bounds=bounds,
                    options={'maxiter': 50, 'gtol': 1e-5}
                )
                warm_start_guess = res_warm.x
                
                # STAGE 1: FoR Hessian (trust-constr)
                # Primary Engine
                res = minimize(
                    fun=lambda p: objective_wrapper(p, L, prev_params),
                    x0=warm_start_guess,
                    method='trust-constr',
                    jac=True,
                    hess=lambda p: hessian_wrapper(p, L, prev_params),
                    bounds=bounds,
                    options={'maxiter': 3000, 'verbose': 0, 'gtol': 1e-13, 'xtol': 1e-13}
                )
                
                # STAGE 2: Safety Fallback (Optimization "Airbag")
                # If Hessian hits a wall (rate approx 0) or fails, try L-BFGS-B
                rate = -res.fun
                if rate < 1e-20:
                     # Fallback to robust gradient descent
                     # print(f"   [Fallback] L={L:.1f} Hessian failed (R={rate:.2e}). Engaging L-BFGS-B.")
                     res = minimize(
                        fun=lambda p: objective_wrapper(p, L, prev_params),
                        x0=current_guess, # Retry from start, or from res.x? Retry from start is safer if res.x is bad.
                        method='L-BFGS-B',
                        jac=True,
                        bounds=bounds,
                        options={'maxiter': 5000, 'gtol': 1e-13}
                    )
                     rate = -res.fun

                best_params = res.x
                
                # SANITY CHECK: Key Rate > 1.0 is physically impossible.
                # If optimization found a numerical hole (rate >> 1), discard it.
                if rate > 1.0:
                    # print(f"   [Warning] Discarding unphysical Key Rate: {rate:.2e} > 1.0")
                    rate = 0.0

                if rate > 1e-20:
                    current_guess = best_params.copy()
                    prev_params = best_params.copy() # Update for next step regularization
                else:
                    # If failed, keep previous guess but record 0. 
                    rate = 0.0

                # STAGE 3: Tertiary Fallback (Dual Annealing) for Low n_X
                # Hessian/L-BFGS-B struggle with the noisy landscape at 10^4.
                # If they fail, we use Dual Annealing to find the global optimum.
                if rate < 1e-20 and n_X <= 1e4:
                    from scipy.optimize import dual_annealing
                    print(f"   [Fallback] L={L:.1f} Gradient methods failed. Engaging Dual Annealing (Stage 3).")
                    
                    def da_wrapper(p):
                        # 1. Enforce Bounds (Clip)
                        # DA respects bounds, but small numerical noise can push slightly outside.
                        # We re-clip to be safe.
                        p_clipped = np.clip(p, [b[0] for b in bounds], [b[1] for b in bounds])
                        
                        # 2. Enforce Sum Constraint (P_mu_1 + P_mu_2 <= 1.0)
                        # If violated, return large penalty (0 rate)
                        if p_clipped[2] + p_clipped[3] >= 0.9999: # P_mu_1 + P_mu_2
                            return 0.0 # Effectively -0.0 key rate

                        # 3. Calculate Objective
                        # DA minimizes, so we return -key_rate
                        try:
                             # scalar_objective returns -KeyRate (negative value)
                             neg_key_rate = scalar_objective(p_clipped, L, n_X, *constants)
                             
                             # 4. Cap Key Rate at 1.0 (Physical Limit)
                             # Calculate actual positive key rate to check
                             real_key_rate = -float(neg_key_rate)
                             
                             # DEBUG: Print high rates
                             if real_key_rate > 1000:
                                print(f"DEBUG: High Rate detected! Val: {neg_key_rate}, Real: {real_key_rate}")
                             
                             # If key rate > 1.0, it's numerical garbage.
                             if real_key_rate > 1.0: 
                                 return 0.0 # Return 0.0 cost (bad result)
                             
                             # Otherwise return the negative key rate (for minimization)
                             return float(neg_key_rate)
                        except Exception as e:
                             print(f"DEBUG: DA Exception: {e}")
                             return 0.0

                    res = dual_annealing(
                        func=da_wrapper,
                        bounds=bounds,
                        maxiter=1000, # Sufficient for low dim
                        no_local_search=True, # Disable L-BFGS-B local search as it might break constraints again
                    )
                    rate = -res.fun
                    if rate > 1e-20:
                        best_params = res.x
                        current_guess = best_params.copy()
                        prev_params = best_params.copy()

                
                # Store by L for merging
                L_key = round(float(L), 4)
                sweep_results[L_key] = (rate, best_params)
                
            except Exception as e:
                print(f"❌ Error at L={L:.1f}, n_X={n_X:.0e}: {e}")
                L_key = round(float(L), 4)
                sweep_results[L_key] = (0.0, start_guess)
        
        return sweep_results

    start_time = time.time()
    
    # 1. FORWARD PASS (0 -> 200)
    print(f"   --> Forward Pass (0-200km) for n_X={n_X:.0e}...")
    forward_results = run_sweep(L_values, initial_guess)
    
    # 2. REVERSE PASS (200 -> 0) - Adaptive Homotopy
    # We only run the reverse sweep for small block sizes (n_X <= 1e5) 
    # where the landscape is noisy and the "systems intelligence" backfill is needed.
    # For large n_X, the forward sweep is rock-solid and we save 50% time by skipping reverse.
    if n_X <= 1e5:
        # Find a good starting point from the end of forward pass (tail is usually stable)
        # Check last 10 points for a valid rate
        tail_guess = initial_guess # Default fallback
        sorted_L = sorted(L_values)
        for L in reversed(sorted_L[-20:]):
            k = round(float(L), 4)
            if k in forward_results and forward_results[k][0] > 1e-15:
                tail_guess = forward_results[k][1]
                break
                
        print(f"   <-- Reverse Pass (200-0km) for n_X={n_X:.0e}...")
        reverse_results = run_sweep(list(reversed(L_values)), tail_guess)
    else:
        print(f"   [Adaptive] Skipping Reverse Pass for n_X={n_X:.0e} (Stable Regime)")
        reverse_results = {}
    
    # 3. MERGE (Stability-Aware)
    # Picks the winner but prefers continuity if rates are very close.
    merged_sequence = []
    prev_merger_params = initial_guess.copy()
    
    for L in L_values:
        k = round(float(L), 4)
        f_rate, f_params = forward_results.get(k, (0.0, np.zeros(5)))
        r_rate, r_params = reverse_results.get(k, (0.0, np.zeros(5)))
        
        # Stability Bias: If rates are very close (within 0.1%), prefer the point 
        # that is closer to the previous point in param space to avoid jumpy basin switches.
        if f_rate > 1e-25 and r_rate > 1e-25:
             ratio = max(f_rate, r_rate) / min(f_rate, r_rate)
             if ratio < 1.001: # Within 0.1%
                  # Pick the one closer to prev point
                  d_f = np.sum((f_params - prev_merger_params)**2)
                  d_r = np.sum((r_params - prev_merger_params)**2)
                  if d_f <= d_r:
                       best_rate, best_params = f_rate, f_params
                  else:
                       best_rate, best_params = r_rate, r_params
             else:
                  # Significant difference, pick the winner normally
                  if f_rate >= r_rate:
                       best_rate, best_params = f_rate, f_params
                  else:
                       best_rate, best_params = r_rate, r_params
        else:
             # One or both are effectively 0, pick the winner normally
             if f_rate >= r_rate:
                  best_rate, best_params = f_rate, f_params
             else:
                  best_rate, best_params = r_rate, r_params
        
        if best_rate > 1e-20:
             prev_merger_params = best_params.copy()
             
        merged_sequence.append({'L': L, 'rate': best_rate, 'params': best_params})

    # 4. STAGE 4: MANIFOLD POLISH (High-Stability Smoothing Pass)
    # This "irons out" residual discontinuities using strong continuity regularization.
    print(f"   ~~> Final Manifold Polish for n_X={n_X:.0e}...")
    final_sequence = []
    current_polish_params = initial_guess.copy()
    lambda_polish = 1e-4 # Stronger regularization for the polish pass
    
    for i, entry in enumerate(merged_sequence):
        L = entry['L']
        r_merged = entry['rate']
        p_merged = entry['params']
        
        if r_merged < 1e-15:
             # Deep in the noise floor, keep the merged result
             final_sequence.append((L, n_X, r_merged, p_merged, initial_guess))
             continue
             
        try:
             # Final L-BFGS-B polish seeded from merged result
             res_polish = minimize(
                 fun=lambda p: objective_wrapper(p, L, current_polish_params),
                 x0=p_merged,
                 method='L-BFGS-B',
                 jac=True,
                 bounds=bounds,
                 options={'maxiter': 500, 'gtol': 1e-14, 'ftol': 1e-18}
             )
             
             if -res_polish.fun > 1e-20:
                  best_rate = -res_polish.fun
                  best_params = res_polish.x
             else:
                  best_rate = r_merged
                  best_params = p_merged
        except:
             best_rate = r_merged
             best_params = p_merged
             
        if best_rate > 1e-20:
             current_polish_params = best_params.copy()
             
        final_sequence.append((L, n_X, best_rate, best_params, initial_guess))

    duration = time.time() - start_time
    return final_sequence, duration

def save_detailed_plot(data, target_nx, output_dir):
    """Generates and saves a detailed plot for a specific n_X."""
    target_nx_str = str(float(target_nx))
    if target_nx_str not in data: return

    filtered_data = [e for e in data[target_nx_str] if e["key_rate"] > 1e-20 and e["key_rate"] < 1.0]
    if not filtered_data: return

    fiber_lengths = [e["fiber_length"] for e in filtered_data]
    key_rates = [e["key_rate"] for e in filtered_data]
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(fiber_lengths, np.log10(key_rates), 'b-', label="Key Rate")
    plt.xlabel("Fiber Length (km)")
    plt.ylabel("log10(Key Rate)")
    plt.title(f"Key Rate (n_X={float(target_nx):.0e})")
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    params = filtered_data[0]["optimized_params"].keys()
    for p in params:
        vals = [e["optimized_params"][p] for e in filtered_data]
        plt.plot(fiber_lengths, vals, label=p)
    plt.xlabel("Fiber Length (km)")
    plt.ylabel("Value")
    plt.title(f"Parameters")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    filename = f"qkd_detailed_nx_{float(target_nx):.0e}.png"
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

def main():
    # --- SETUP RUN DIRECTORY ---
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # Output to Hessian_FoR
    run_dir = os.path.join(script_dir, "../outputs/JAX/Hessian_FoR", f"run_HESSIAN_FoR_{timestamp}")
    detailed_plots_dir = os.path.join(run_dir, "detailed_plots")
    os.makedirs(detailed_plots_dir, exist_ok=True)
    
    # --- SAVE METADATA ---
    metadata = {
        "script_name": os.path.basename(__file__),
        "strategy": "Hessian FoR + Warming",
        "timestamp": timestamp,
        "description": "Systems Intelligence Architecture: Stage 1 = FoR Hessian (Trust-Constr) + Tikhonov Regularization. Stage 2 = L-BFGS-B Safety Fallback. Stage 3 = Dual Annealing (Rare Fallback for n_X=10^4). Adaptive Homotopy: Bi-Directional scan for n_X <= 1e5, Forward-only for n_X >= 1e6."
    }
    with open(os.path.join(run_dir, "run_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"🚀 Starting FoR HESSIAN + WARMING Optimization Run: {timestamp}")
    print(f"📂 Output Directory: {run_dir}")

    # --- SIMULATION PARAMETERS ---
    L_values = np.linspace(0, 200, 400) # 0.5km steps
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

    bounds = [(4e-4, 0.9), (2e-4, 0.5), (1e-12, 0.999), (1e-12, 0.999), (1e-12, 0.999)]
    
    # Initial Guesses
    initial_guesses_map = {
        1e4: np.array([0.65, 0.15, 0.05, 0.61, 0.425]),
        1e9: np.array([0.52, 0.40, 0.18, 0.785, 0.88]),
    }
    default_guess = np.array([0.52, 0.40, 0.18, 0.785, 0.88])

    print(f"🏎️  Running optimization...")
    wall_start = time.time()
    total_sequential_time = 0.0
    timing_data = {}
    final_results = []
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_nx = {}
        for n_X in n_X_values:
            guess = initial_guesses_map.get(n_X, default_guess)
            future = executor.submit(optimize_single_nx_sequence_for, n_X, L_values, bounds, guess, constants)
            future_to_nx[future] = n_X
            
        for future in concurrent.futures.as_completed(future_to_nx):
            n_X = future_to_nx[future]
            try:
                results, duration = future.result()
                final_results.extend(results)
                timing_data[str(float(n_X))] = duration
                total_sequential_time += duration
                print(f"✅ Finished n_X={n_X:.0e} in {duration/60:.2f} min")
            except Exception as e:
                print(f"❌ Exception for n_X={n_X}: {e}")

    wall_end = time.time()
    wall_duration = wall_end - wall_start
    print(f"\n🏁 Optimization Complete in {wall_duration/60:.2f} min (Wall Time)")

    # --- SAVE RESULTS ---
    raw_dataset = [{
        "fiber_length": float(r[0]),
        "n_X": int(r[1]),
        "key_rate": float(r[2]),
        "optimized_parameters": {
            "mu_1": float(r[3][0]), "mu_2": float(r[3][1]), "P_mu_1": float(r[3][2]), 
            "P_mu_2": float(r[3][3]), "P_X_value": float(r[3][4])
        }
    } for r in final_results]
    
    grouped_data = {}
    for entry in raw_dataset:
        n_X = float(entry["n_X"])
        entry["e_1"] = entry["fiber_length"] / 100
        entry["e_2"] = -np.log10(P_dc_value)
        entry["e_3"] = e_mis * 100
        entry["e_4"] = np.log10(n_X)
        entry["optimized_params"] = entry["optimized_parameters"]
        
        nx_str = str(n_X)
        if nx_str not in grouped_data: grouped_data[nx_str] = []
        grouped_data[nx_str].append(entry)
    
    for nx in grouped_data: grouped_data[nx].sort(key=lambda x: x["fiber_length"])
        
    grouped_path = os.path.join(run_dir, f"qkd_grouped_dataset_{timestamp}.json")
    with open(grouped_path, 'w') as f:
        json.dump(grouped_data, f, indent=2)

    try: generate_timing_report.generate_report(timing_data, total_sequential_time, wall_duration, os.cpu_count(), run_dir)
    except: pass

    for nx in grouped_data.keys():
        try: save_detailed_plot(grouped_data, nx, detailed_plots_dir)
        except: pass
            
    print(f"\n✅ FoR Hessian Run Finished. Results in: {run_dir}")

if __name__ == "__main__":
    main()
