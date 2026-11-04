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
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu") # Force CPU for 64-bit precision

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- JIT COMPILE HESSIAN ---
# We compute the Hessian of the scalar objective function
hessian_func = jax.jit(jax.hessian(scalar_objective))

def optimize_single_nx_sequence_hessian(n_X, L_values, bounds, initial_guess, constants):
    """
    Optimizes key rate using Trust-Region Newton-CG (trust-ncg) with exact Hessian.
    """
    alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event = constants
    
    results = []
    current_best_guess = initial_guess.copy()
    
    # Wrapper for minimize (Video/Gradient)
    def objective_wrapper(params, L_val):
        val, grad = objective_val_and_grad(
            params, L_val, n_X, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event
        )
        if not np.isfinite(val): return 1e9, np.zeros_like(params)
        # grad = np.clip(np.array(grad), -1e6, 1e6) # Clipping might confuse trust-ncg, removing for now or keep mild?
        return float(val), np.array(grad)

    # Wrapper for Hessian
    def hessian_wrapper(params, L_val):
        H = hessian_func(
            jnp.array(params), L_val, n_X, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event
        )
        # Ensure numpy array
        return np.array(H)

    start_time = time.time()
    
    for i, L in enumerate(L_values):
        try:
            # For Trust-NCG (and Newton methods in general), good initialization is key.
            # We use the current_best_guess (continuation method).
            
            # Additional safety: trust-ncg doesn't support bounds natively in scipy < 1.5 (wrapper usually handles it or ignores).
            # Actually, standard 'trust-ncg' in scipy DOES NOT support `bounds`.
            # 'trust-constr' supports bounds, but is slower.
            # 'L-BFGS-B' supports bounds.
            # 'Newton-CG' does NOT support bounds.
            
            # WORKAROUND:
            # QKD parameters are bounded [0, 1].
            # We can use 'trust-constr' which supports bounds and Hessian.
            # OR we can purely rely on the fact that the objective function returns infinity outside valid range?
            # No, gradient descent needs bounds or parameter transformation through sigmoid.
            
            # Let's use 'trust-constr' as it supports bounds + Hessian.
            # IT IS SLOWER than Newton-CG but correct.
            
            # ALTERNATIVE: Use "Newton-CG" but manually clip parameters inside the objective? 
            # No, that breaks gradients.
            
            # Given the user asked for "Newton-CG", which is for unconstrained problems mostly.
            # We will use 'trust-constr' for correctness with bounds.
            
            # --- STAGE 1: HESSIAN (Global Search) ---
            # Use trust-constr to navigate the landscape and handle bounds with exact Hessian
            start_time_stage1 = time.time()
            res_stage1 = minimize(
                fun=lambda p: objective_wrapper(p, L),
                x0=current_best_guess,
                method='trust-constr',
                jac=True,
                hess=lambda p: hessian_wrapper(p, L),
                bounds=bounds,
                options={'maxiter': 3000, 'verbose': 0, 'gtol': 1e-13, 'xtol': 1e-13}
            )
            stage1_time = time.time() - start_time_stage1
            
            # --- STAGE 2: ADAPTIVE POLISH (High Precision) ---
            # Use L-BFGS-B starting from the Hessian result to smooth out any jitter.
            # Adaptive: Use more iterations for L > 100km where the landscape is flatter.
            maxiter_polish = 15000 if L > 100 else 3000
            
            start_time_stage2 = time.time()
            res = minimize(
                fun=lambda p: objective_wrapper(p, L),
                x0=res_stage1.x,  # Start from Hessian result
                method='L-BFGS-B',
                jac=True,  # Use exact gradient
                bounds=bounds,
                options={'maxiter': maxiter_polish, 'ftol': 2.22e-16, 'gtol': 1e-14}
            )
            stage2_time = time.time() - start_time_stage2
            
            rate = -res.fun 
            # print(f"  L={L:.1f}km: Stage1={stage1_time:.2f}s, Stage2={stage2_time:.2f}s, Rate={rate:.5e}")
            best_params = res.x
            
            # Check success (trust-constr might stop early)
            if rate > 1e-20:
                current_best_guess = best_params.copy()
            else:
                # If trust-constr fails to find a positive rate (it might be too conservative),
                # we could fallback or just accept 0.
                if L <= 100:
                    # Retry with L-BFGS-B if Hessian method fails early (robustness)
                    # Or just return 0 to show true performance of Hessian method alone?
                    # Let's stick to Hessian method to see its true colors.
                    pass
                best_params = current_best_guess
                rate = 0.0

            results.append((L, n_X, rate, best_params, initial_guess))
            
        except Exception as e:
            print(f"❌ Error at L={L:.1f}, n_X={n_X:.0e}: {e}")
            results.append((L, n_X, 0.0, [0.0]*5, initial_guess))

    duration = time.time() - start_time
    return results, duration

def save_detailed_plot(data, target_nx, output_dir):
    """Generates and saves a detailed plot for a specific n_X."""
    target_nx_str = str(float(target_nx))
    if target_nx_str not in data: return

    filtered_data = [e for e in data[target_nx_str] if e["key_rate"] > 1e-20]
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
    run_dir = os.path.join(script_dir, "../outputs/JAX/Hessian", f"run_HESSIAN_{timestamp}")
    detailed_plots_dir = os.path.join(run_dir, "detailed_plots")
    os.makedirs(detailed_plots_dir, exist_ok=True)
    
    # --- SAVE METADATA ---
    metadata = {
        "script_name": os.path.basename(__file__),
        "strategy": "Hessian + Adaptive Deep Polish",
        "timestamp": timestamp,
        "description": "Hybrid Strategy: Stage 1 = Hessian (Trust-Constr). Stage 2 = Adaptive Deep Polish (L-BFGS-B, maxiter=15000 for L>100km, 3000 else, gtol=1e-14)."
    }
    with open(os.path.join(run_dir, "run_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"🚀 Starting HESSIAN (Trust-Constr) Optimization Run: {timestamp}")
    print(f"📂 Output Directory: {run_dir}")

    # --- SIMULATION PARAMETERS ---
    L_values = np.linspace(0, 200, 400) # 0.5km steps
    n_X_values = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
    # n_X_values = [1e9] # Test Only
    
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
            future = executor.submit(optimize_single_nx_sequence_hessian, n_X, L_values, bounds, guess, constants)
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
            
    print(f"\n✅ Hessian Run Finished. Results in: {run_dir}")

if __name__ == "__main__":
    main()
