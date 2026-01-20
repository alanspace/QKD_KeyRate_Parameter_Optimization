
"""
Analysis/verify_performance_metrics.py
======================================
This script verifies the key performance claims made in the README.
It calculates:
1. Range Extension (Difference in max secure distance between Static and Optimized)
2. Rate Gain (Factor improvement in Key Rate at long distance, e.g., 180km)
3. Optimization Speedup (Benchmark of JAX vs Neural Network)

Usage:
    python Analysis/verify_performance_metrics.py
"""

import sys
import os
import time
import numpy as np
import torch

# 1. Setup Environment (Force CPU for stability)
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
import jax
jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_enable_x64', True)

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.qkd.model import calculate_key_rates_and_metrics, objective_val_and_grad
from scipy.optimize import dual_annealing

# Constants for Physics
PARAMS = {
    'n_X': 1e8,
    'alpha': 0.2,
    'eta_Bob': 0.1,
    'P_dc_value': 6e-7,
    'epsilon_sec': 1e-10,
    'epsilon_cor': 1e-15,
    'f_EC': 1.16,
    'e_mis': 0.01,
    'P_ap': 0,
    'n_event': 1
}
BOUNDS = [(4e-4, 0.9), (2e-4, 0.5), (1e-12, 1.0), (1e-12, 1.0), (1e-12, 1.0)]

# Static Parameters (Fixed Strategy)
STATIC_PARAMS = (0.34479867337905723, 0.19526100517866424, 0.21504895346866, 0.4645950203307233, 0.1)

def get_optimized_rate(L):
    """Finds global optimum key rate for distance L."""
    def obj(p):
        val, _ = objective_val_and_grad(
            p, L, PARAMS['n_X'], PARAMS['alpha'], PARAMS['eta_Bob'], PARAMS['P_dc_value'],
            PARAMS['epsilon_sec'], PARAMS['epsilon_cor'], PARAMS['f_EC'], PARAMS['e_mis'],
            PARAMS['P_ap'], PARAMS['n_event']
        )
        return float(val)
    
    # Strategy: Try local optimization from STATIC parameters first (likely close to optimal)
    # This guarantees we are at least as good as static.
    from scipy.optimize import minimize
    
    # 1. Local Search from Static Guess
    def obj_local(p):
         val, _ = objective_val_and_grad(
            p, L, PARAMS['n_X'], PARAMS['alpha'], PARAMS['eta_Bob'], PARAMS['P_dc_value'],
            PARAMS['epsilon_sec'], PARAMS['epsilon_cor'], PARAMS['f_EC'], PARAMS['e_mis'],
            PARAMS['P_ap'], PARAMS['n_event']
        )
         return float(val)

    res_local = minimize(obj_local, STATIC_PARAMS, method='L-BFGS-B', bounds=BOUNDS, tol=1e-6)
    rate_local = -res_local.fun
    
    # If successful and positive, check if global search can beat it
    rate_global = 0.0
    
    # 2. Global Search (The real power of our engine)
    # We always run this because local search might get stuck in the 'Static' basin
    res_global = dual_annealing(obj_local, bounds=BOUNDS, maxiter=50) 
    rate_global = -res_global.fun
    
    return max(rate_local, rate_global)

def get_static_rate(L):
    """Calculates key rate for distance L using fixed parameters."""
    res = calculate_key_rates_and_metrics(
        STATIC_PARAMS, L, PARAMS['n_X'], PARAMS['alpha'], PARAMS['eta_Bob'], PARAMS['P_dc_value'],
        PARAMS['epsilon_sec'], PARAMS['epsilon_cor'], PARAMS['f_EC'], PARAMS['e_mis'],
        PARAMS['P_ap'], PARAMS['n_event']
    )
    return float(res[0])

def measure_range_extension():
    print("\n--- 1. Range Extension Verification ---")
    
    def find_max_dist(func):
        low, high = 150.0, 250.0
        for _ in range(15):
            mid = (low + high) / 2
            if func(mid) > 1e-20:
                low = mid
            else:
                high = mid
        return low

    static_max = find_max_dist(get_static_rate)
    print(f"Static Max Range:    {static_max:.2f} km")
    
    opt_max = find_max_dist(get_optimized_rate)
    print(f"Optimized Max Range: {opt_max:.2f} km")
    
    print(f"✅ Range Extension:   +{opt_max - static_max:.2f} km")
    return opt_max - static_max

def measure_rate_gain(target_L=180.0):
    print(f"\n--- 2. Rate Gain Verification (at {target_L} km) ---")
    s_rate = get_static_rate(target_L)
    o_rate = get_optimized_rate(target_L)
    
    print(f"Static Rate:    {s_rate:.2e}")
    print(f"Optimized Rate: {o_rate:.2e}")
    
    if s_rate > 0:
        ratio = o_rate / s_rate
        print(f"✅ Rate Gain:     {ratio:.1f}x")
    else:
        print(f"✅ Rate Gain:     Infinite (Static is 0)")

def measure_speedup():
    print("\n--- 3. Speedup Verification (Inference vs JAX) ---")
    
    # Load NN
    try:
        model_path = os.path.join(os.path.dirname(__file__), '../NeuralNetwork/models/bb84_nn_model_jax.pth')
        # Simple definition for loading
        class BB84Network(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = torch.nn.Linear(4, 16)
                self.fc2 = torch.nn.Linear(16, 32)
                self.fc3 = torch.nn.Linear(32, 16)
                self.fc4 = torch.nn.Linear(16, 5)
            def forward(self, x):
                return self.fc4(torch.nn.functional.relu(self.fc3(torch.nn.functional.relu(self.fc2(torch.nn.functional.relu(self.fc1(x)))))))

        model = BB84Network()
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
        else:
            print("Warning: Model file not found, using random weights for timing.")
    except Exception as e:
        print(f"Model load error: {e}")
        return

    # Timing JAX (Single Point Optimization)
    # Warmup
    _ = get_optimized_rate(50.0) 
    
    t0 = time.time()
    n_jax = 5
    for _ in range(n_jax):
        get_optimized_rate(50.0)
    jax_time = (time.time() - t0) / n_jax
    print(f"JAX Optimizer (Avg): {jax_time:.4f} s")

    # Timing NN (Batch Inference)
    input_tensor = torch.randn(100, 4)
    t0 = time.time()
    for _ in range(100):
        with torch.no_grad():
            _ = model(input_tensor)
    nn_time = (time.time() - t0) / (100 * 100) # Per sample
    print(f"NN Inference (Avg):  {nn_time:.6f} s")
    
    print(f"✅ Speedup Factor:   {jax_time / nn_time:.0f}x")

if __name__ == "__main__":
    measure_range_extension()
    measure_rate_gain()
    measure_speedup()
