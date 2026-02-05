
import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
from jax import config

# Enable NaN debugging
config.update("jax_debug_nans", True)
config.update("jax_platform_name", "cpu")
config.update("jax_enable_x64", True)

# Ensure project root is in path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.qkd.model import objective, scalar_objective

def reproduce():
    print("🚀 Starting NaN Reproduction...")
    
    # Constants
    L_values = np.linspace(0, 200, 200) # Use full array as in real script
    n_X = 1e4
    
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
    
    # Bounds
    bounds = [(4e-4, 0.9), (2e-4, 0.5), (1e-6, 1.0 - 1e-6), (1e-6, 1.0 - 1e-6), (1e-6, 1.0 - 1e-6)]
    
    # Generate random points until we hit a NaN
    rng = np.random.default_rng(42)
    
    for i in range(100):
        # Generate valid candidate
        while True:
            cand = rng.uniform(
                low=[b[0] for b in bounds], 
                high=[b[1] for b in bounds], 
                size=(5,)
            )
            # Mu params: mu_1 (0), mu_2 (1). Prob params: P_mu_1 (2), P_mu_2 (3), P_X (4)
            if (cand[2] + cand[3] < 0.999) and \
               (cand[0] > cand[1] + 2e-4) and \
               (cand[1] > 2e-4):
                params = jnp.array(cand)
                break
        
        print(f"Testing params {i}: {params}")
        
        try:
            # Calculate Value and Gradient
            # scalar_objective uses jax.jit, so jax_debug_nans should catch it inside
            val_loss = scalar_objective(params, L_values, n_X, *constants)
            print(f"Val: {val_loss}")
            
            grad_fn = jax.grad(scalar_objective)
            grads = grad_fn(params, L_values, n_X, *constants)
            print(f"Grads: {grads}")
            
        except Exception as e:
            print(f"\n💥 CAUGHT EXCEPTION AT ITERATION {i}")
            print(e)
            # Re-raise to see traceback if needed, but printing e is usually enough with debug_nans
            raise e

if __name__ == "__main__":
    reproduce()
