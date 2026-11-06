#!/usr/bin/env bash
# Leonardo login node: physics env for the main project (CPU, float64).
set -euo pipefail
cd "$(dirname "$0")/../.."
module load python 2>/dev/null || true
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt "jax[cpu]==0.9.0"
JAX_PLATFORMS=cpu .venv/bin/python -c "
import jax; jax.config.update('jax_enable_x64', True)
from src.qkd_opt import get_preset; from src.qkd_opt.optimize import optimize
r = optimize(100.0, 1e9, get_preset('lim')); print('self-test R(100 km, 1e9) =', float(r['R']), '(expected ~8.537e-05)')"
