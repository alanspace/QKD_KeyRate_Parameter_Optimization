import time
import sys
import os
import numpy as np

sys.path.append('/Users/daai6ga1hou2/Documents/GitHub/Physics/QKD_KeyRate_Parameter_Optimization')
os.environ['JAX_PLATFORMS'] = 'cpu'

from src.qkd.model import objective_val_and_grad
from scipy.optimize import dual_annealing

PHYSICS_CONSTANTS = {
    'alpha': 0.2, 'eta_Bob': 0.1, 'P_dc_value': 6e-7, 
    'epsilon_sec': 1e-10, 'epsilon_cor': 1e-15, 'f_EC': 1.16, 
    'e_mis': 5e-3, 'P_ap': 0, 'n_event': 1
}

def obj(p):
    return float(objective_val_and_grad(np.array(p), 100.0, 1e8, **PHYSICS_CONSTANTS)[0])

start = time.time()
res = dual_annealing(obj, bounds=[(4e-4, 0.9), (2e-4, 0.5), (1e-12, 1.0), (1e-12, 1.0), (1e-12, 1.0)], maxiter=1000)
print(f"Time for 1000 iter 5D optimization (JAX backend): {time.time()-start:.2f} seconds")
