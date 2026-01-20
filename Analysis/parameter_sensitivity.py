
"""
Analysis/parameter_sensitivity.py
=================================
This script analyzes the sensitivity of the Key Rate to individual parameters.
It answers: "Which parameter is most critical to optimize?"

Methodology:
1. Find Global Optimum at a reference distance (L=100km).
2. For each parameter (mu1, mu2, Px, etc.):
   - Deviate it by +/- 20% from optimal.
   - Re-optimize all OTHER parameters.
   - Measure the Key Rate drop.
3. Plot the results as a Sensitivity Bar Chart.

Usage:
    python Analysis/parameter_sensitivity.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize

# Force CPU JAX
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
import jax
jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_enable_x64', True)

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.qkd.model import calculate_key_rates_and_metrics, objective_val_and_grad

# Constants
L_TARGET = 100.0 # km
NX = 1e8
PARAMS_FIXED = {
    'alpha': 0.2, 'eta_Bob': 0.1, 'P_dc_value': 6e-7, 'epsilon_sec': 1e-10,
    'epsilon_cor': 1e-15, 'f_EC': 1.16, 'e_mis': 0.01, 'P_ap': 0, 'n_event': 1
}
BOUNDS = [(4e-4, 0.9), (2e-4, 0.5), (1e-12, 1.0), (1e-12, 1.0), (1e-12, 1.0)]

def get_optimum(frozen_idx=None, frozen_val=None):
    """
    Finds optimum key rate.
    If frozen_idx is set, that parameter is held constant at frozen_val.
    """
    def obj(p):
        # Insert frozen parameter if needed
        p_full = list(p)
        if frozen_idx is not None:
            p_full.insert(frozen_idx, frozen_val)
        
        val, _ = objective_val_and_grad(
            p_full, L_TARGET, NX, **PARAMS_FIXED
        )
        return float(val)

    # Adjust bounds/x0 if one param is removed
    current_bounds = list(BOUNDS)
    x0 = [0.5, 0.1, 0.5, 0.5, 0.5]
    
    
    if frozen_idx is not None:
        del current_bounds[frozen_idx]
        del x0[frozen_idx]
    
    # Use Dual Annealing for robustness (especially for Baseline)
    # We want to be sure we are at the Global Optimum before creating sensitivity relative to it.
    from scipy.optimize import dual_annealing
    res = dual_annealing(obj, bounds=current_bounds, maxiter=50)
    return -res.fun, res.x

def run_sensitivity_analysis():
    print(f"🚀 Running Sensitivity Analysis at L={L_TARGET}km...")
    
    # 1. Baseline Optimization (Control)
    base_rate, base_params = get_optimum()
    print(f"✅ Baseline Rate: {base_rate:.2e} (Params: {np.round(base_params, 3)})")
    
    param_names = ['Signal (μ1)', 'Decoy (μ2)', 'Prob (P_μ1)', 'Prob (P_μ2)', 'Basis (Px)']
    results = []
    
    # 2. Perturbation Test
    # For each parameter, we deviate it by 20% and see how much rate we lose
    # even if we re-optimize everything else.
    deviation = 0.20 # 20%
    
    for i, name in enumerate(param_names):
        optimal_val = base_params[i]
        
        # Test +20%
        val_high = optimal_val * (1 + deviation)
        # Clamp to bounds
        val_high = min(val_high, BOUNDS[i][1])
        rate_high, _ = get_optimum(frozen_idx=i, frozen_val=val_high)
        
        # Test -20%
        val_low = optimal_val * (1 - deviation)
        val_low = max(val_low, BOUNDS[i][0])
        rate_low, _ = get_optimum(frozen_idx=i, frozen_val=val_low)
        
        loss_high = (base_rate - rate_high) / base_rate * 100
        loss_low = (base_rate - rate_low) / base_rate * 100
        
        max_loss = max(loss_high, loss_low)
        print(f"   - {name}: Sensitive? Loss = {max_loss:.1f}%")
        
        results.append({
            'Parameter': name,
            'Sensitivity (%)': max_loss
        })

    # 3. Plotting
    df = pd.DataFrame(results).sort_values(by='Sensitivity (%)', ascending=True)
    
    plt.figure(figsize=(10, 6))
    bars = plt.barh(df['Parameter'], df['Sensitivity (%)'], color='#ff7f0e')
    plt.xlabel('Key Rate Loss (%) on 20% Mis-calibrated Parameter')
    plt.title(f'Parameter Sensitivity Analysis (L={L_TARGET}km)\n"Which knob matters most?"')
    plt.grid(True, axis='x', linestyle='--', alpha=0.5)
    
    # Add value labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
                 f'{width:.1f}%', va='center', fontweight='bold')
        
    out_dir = os.path.join(os.path.dirname(__file__), '../Testing')
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    save_path = os.path.join(out_dir, "Sensitivity_Analysis.png")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"✅ Saved plot to {save_path}")

if __name__ == "__main__":
    run_sensitivity_analysis()
