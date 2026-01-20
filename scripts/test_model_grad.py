import os
import sys
import numpy as np
import jax
import jax.numpy as jnp

sys.path.append(os.getcwd())
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

from src.qkd.model import objective_val_and_grad, calculate_key_rates_and_metrics, scalar_objective
from jax import value_and_grad

# Parameters
L = 0.0
n_X = 1e9
alpha = 0.2
eta_Bob = 0.1
P_dc_value = 6e-7
epsilon_sec = 1e-10
epsilon_cor = 1e-15
f_EC = 1.16
e_mis = 0.01
P_ap = 0
n_event = 1
x0 = np.array([0.52, 0.40, 0.18, 0.785, 0.88])

print(f"Testing L={L} with params={x0}")

# 1. Test standard calculation (no grad)
print("Running calculate_key_rates_and_metrics...")
# Ensure metrics run also in non-JIT if needed, but model's func is decorated.
# We will use disable_jit context for EVERYTHING.

with jax.disable_jit():
    metrics = calculate_key_rates_and_metrics(x0, L, n_X, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event)
    key_rate = metrics[0]
    print(f"Raw Key Rate (metrics[0]): {key_rate}")

    # 2. Test val_and_grad
    print("Running objective_val_and_grad NO JIT...")
    objective_val_and_grad_nojit = value_and_grad(scalar_objective, argnums=0)
    
    try:
        val, grad = objective_val_and_grad_nojit(x0, L, n_X, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event)
        print(f"Objective Value: {val}")
        print(f"Gradient: {grad}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

