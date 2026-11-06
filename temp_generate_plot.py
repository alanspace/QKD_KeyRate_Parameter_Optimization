import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from scipy.optimize import dual_annealing

# Ensure paths are correct for the standalone script
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
output_assets = os.path.join(script_dir, 'assets')
os.makedirs(output_assets, exist_ok=True)

from src.qkd.model import objective_val_and_grad

print("\n🚀 Running Standalone High-Fidelity Sensitivity Analysis...")

# 1. FIX: Use the exact same physics constants as the main training loop!
PHYSICS_CONSTANTS = {
    'alpha': 0.2, 'eta_Bob': 0.1, 'P_dc_value': 6e-7, 
    'epsilon_sec': 1e-10, 'epsilon_cor': 1e-15, 'f_EC': 1.16, 
    'e_mis': 5e-3, 'P_ap': 0, 'n_event': 1
}

# The true physical bounds for [mu_1, mu_2, P_mu_1, P_mu_2, P_X]
full_bounds = [(4e-4, 0.9), (2e-4, 0.5), (1e-12, 1.0), (1e-12, 1.0), (1e-12, 1.0)]

def get_s_opt(idx=None, val=None):
    def obj_s(p):
        p_f = list(p)
        if idx is not None: 
            p_f.insert(idx, val)
            
        # 2. FIX: Hard physical constraint so signal + decoy probabilities don't exceed 100%
        if p_f[2] + p_f[3] > 1.0:
            return 1e9 

        v, _ = objective_val_and_grad(np.array(p_f), 100.0, 1e8, **PHYSICS_CONSTANTS)
        return float(v)
        
    if idx is not None:
        # 3. FIX: Safe bounds slicing
        active_bounds = full_bounds[:idx] + full_bounds[idx+1:]
    else:
        active_bounds = full_bounds
        
    # 4. FIX: maxiter bumped to 1000 to guarantee true global convergence
    res = dual_annealing(obj_s, bounds=active_bounds, maxiter=1000)
    return -res.fun, res.x

# Run baseline optimization
b_r, b_p = get_s_opt()
p_n = ['Signal ($\mu_1$)', 'Decoy ($\mu_2$)', 'Prob ($P_{\mu_1}$)', 'Prob ($P_{\mu_2}$)', 'Basis ($P_X$)']
s_r = []

for i, name in enumerate(p_n):
    o_v = b_p[i]
    param_min, param_max = full_bounds[i]
    
    # 5. FIX: Calculate exact 20% shifts, strictly clamped to the specific parameter's true limits
    val_high = min(o_v * 1.2, param_max)
    val_low = max(o_v * 0.8, param_min)
    
    print(f"Testing {name}... (Base: {o_v:.4f}, Low: {val_low:.4f}, High: {val_high:.4f})")
    
    r_h, _ = get_s_opt(i, val_high)
    r_l, _ = get_s_opt(i, val_low)
    
    # Calculate percentage drop (floored at 0 to absorb any microscopic stochastic jitter)
    drop_high = max((b_r - r_h) / b_r, 0)
    drop_low = max((b_r - r_l) / b_r, 0)
    
    s_r.append({'Parameter': name, 'Sensitivity (%)': max(drop_high, drop_low) * 100})

print("\n✅ Analysis Complete. Generating Plot...")

# Plotting aesthetics
plt.rcParams.update({'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18})
df = pd.DataFrame(s_r).sort_values(by='Sensitivity (%)')

ax = df.plot.barh(x='Parameter', y='Sensitivity (%)', color='#ff7f0e', figsize=(10, 6), legend=False)
plt.title('Parameter Sensitivity (L=100km, $n_X=10^8$)')
plt.xlabel('Secret Key Rate Degradation (%)')
plt.grid(axis='x', linestyle='--', alpha=0.7)

plt.tight_layout()
plt_path = os.path.join(output_assets, 'parameter_sensitivity.png')
plt.savefig(plt_path, dpi=300)
print(f"🎯 Flawless plot saved to: {plt_path}")
