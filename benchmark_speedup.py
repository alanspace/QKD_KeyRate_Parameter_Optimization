#!/usr/bin/env python3
"""
Quick benchmark script to verify speedup claims.
Measures:
1. JAX optimizer speed (single point)
2. Neural Network inference speed (single point)
3. Calculates actual speedup
"""

import sys
import os
import time
import numpy as np
import torch

# Add project root
sys.path.append(os.getcwd())

# JAX Configuration
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
import jax
jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_enable_x64', True)

from src.qkd.model import objective_val_and_grad
from scipy.optimize import minimize, dual_annealing

# Test parameters
PARAMS = {
    'L': 50.0,  # 50 km
    'n_X': 1e9,
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
x0 = np.array([0.52, 0.40, 0.18, 0.785, 0.88])

def benchmark_jax_optimizer(n_runs=10):
    """Benchmark JAX-based optimizer."""
    print("\n=== Benchmarking JAX Optimizer ===")
    
    def objective_wrapper(params):
        val, grad = objective_val_and_grad(
            params, PARAMS['L'], PARAMS['n_X'], PARAMS['alpha'], 
            PARAMS['eta_Bob'], PARAMS['P_dc_value'], PARAMS['epsilon_sec'], 
            PARAMS['epsilon_cor'], PARAMS['f_EC'], PARAMS['e_mis'], 
            PARAMS['P_ap'], PARAMS['n_event']
        )
        return float(val), np.array(grad)
    
    # Warmup
    print("Warming up...")
    result = minimize(
        fun=objective_wrapper,
        x0=x0,
        method='L-BFGS-B',
        jac=True,
        bounds=BOUNDS,
        options={'maxiter': 200, 'ftol': 1e-12}
    )
    
    # Actual benchmark
    print(f"Running {n_runs} optimization iterations...")
    times = []
    for i in range(n_runs):
        start = time.perf_counter()
        result = minimize(
            fun=objective_wrapper,
            x0=x0,
            method='L-BFGS-B',
            jac=True,
            bounds=BOUNDS,
            options={'maxiter': 200, 'ftol': 1e-12}
        )
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed*1000:.2f} ms")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"\n📊 JAX Optimizer Results:")
    print(f"  Average: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
    print(f"  Median:  {np.median(times)*1000:.2f} ms")
    print(f"  Min:     {np.min(times)*1000:.2f} ms")
    print(f"  Max:     {np.max(times)*1000:.2f} ms")
    
    return avg_time

def benchmark_nn_inference(n_runs=10000):
    """Benchmark Neural Network inference."""
    print("\n=== Benchmarking Neural Network ===")
    
    # Load model
    model_path = 'NeuralNetwork/models/bb84_nn_model_jax.pth'
    
    class BB84Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(4, 16)
            self.fc2 = torch.nn.Linear(16, 32)
            self.fc3 = torch.nn.Linear(32, 16)
            self.fc4 = torch.nn.Linear(16, 5)
        
        def forward(self, x):
            x = torch.nn.functional.relu(self.fc1(x))
            x = torch.nn.functional.relu(self.fc2(x))
            x = torch.nn.functional.relu(self.fc3(x))
            return self.fc4(x)
    
    model = BB84Network()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"✅ Loaded model from {model_path}")
    else:
        print(f"⚠️  Model not found at {model_path}, using random weights")
    
    model.eval()
    
    # Create sample input
    sample_input = torch.tensor([[PARAMS['L'], PARAMS['P_dc_value'], 
                                  PARAMS['e_mis'], PARAMS['n_X']]], 
                                dtype=torch.float32)
    
    # Warmup
    with torch.no_grad():
        for _ in range(100):
            _ = model(sample_input)
    
    # Benchmark
    print(f"Running {n_runs} inference iterations...")
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(sample_input)
    total_time = time.perf_counter() - start
    avg_time = total_time / n_runs
    
    print(f"\n📊 Neural Network Results:")
    print(f"  Total time: {total_time*1000:.2f} ms for {n_runs} inferences")
    print(f"  Average:    {avg_time*1e6:.4f} microseconds per inference")
    print(f"  Throughput: {n_runs/total_time:.0f} inferences/second")
    
    return avg_time

def main():
    print("🚀 QKD Optimization Speed Benchmark")
    print("=" * 60)
    
    # Benchmark JAX optimizer
    jax_time = benchmark_jax_optimizer(n_runs=10)
    
    # Benchmark NN
    nn_time = benchmark_nn_inference(n_runs=10000)
    
    # Calculate speedup
    print("\n" + "=" * 60)
    print("📈 SPEEDUP ANALYSIS")
    print("=" * 60)
    
    speedup = jax_time / nn_time
    
    print(f"\n1️⃣  JAX Optimizer (L-BFGS-B):  {jax_time*1000:.2f} ms per point")
    print(f"2️⃣  Neural Network Inference:   {nn_time*1e6:.4f} μs per point")
    print(f"\n🚀 SPEEDUP: {speedup:,.0f}x")
    
    # Context
    print(f"\n💡 What this means:")
    print(f"   - For 1000 fiber lengths, JAX takes: {jax_time*1000:.1f} seconds")
    print(f"   - For 1000 fiber lengths, NN takes:  {nn_time*1000:.3f} seconds")
    print(f"   - Time saved: {(jax_time - nn_time)*1000:.1f} seconds per 1000 points")
    
    # Compare to README claims
    print(f"\n📋 Comparison to README Claims:")
    print(f"   - README claims: >6,000x (conservative)")
    print(f"   - Actual measured: {speedup:,.0f}x")
    
    if speedup >= 6000:
        print(f"   ✅ Claim VERIFIED (exceeds minimum)")
    elif speedup >= 4000:
        print(f"   ⚠️  Close but slightly under (still impressive)")
    else:
        print(f"   ❌ Significantly different from claim")

if __name__ == "__main__":
    main()
