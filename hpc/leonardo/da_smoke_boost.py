# Booster CPU smoke test of the archived DA pipeline: 3 points, one core, output only in results/da_smoke_boost/.
# Run on Leonardo 2026-11-08 from the repo root (PYTHONPATH=$PWD, JAX_PLATFORMS=cpu; sbatch --wrap, job 63915182);
# job 63912032 failed on an f-string bug in an earlier copy of this file. Copied here from the Leonardo clone.
# Usage (repo root): PYTHONPATH=. JAX_PLATFORMS=cpu python hpc/leonardo/da_smoke_boost.py
import json, os, sys, time
sys.argv = [sys.argv[0]]
sys.path.insert(0, "Optimization/benchmark")
import da_dataset as D
import jax.numpy as jnp
from src.qkd_opt import get_preset, check_x64
check_x64()
f, vg = D.TB._objectives(get_preset("lim"))
t = time.perf_counter(); f(jnp.asarray(D.TB.X0), 120.0, 1e9).block_until_ready(); vg(jnp.asarray(D.TB.X0), 50.0, 1e9)
print(f"warm-up (JIT compile) {time.perf_counter() - t:.1f}s", flush=True)
rows = []
for i, (L, nx) in enumerate([(50.0, 1e9), (130.0, 1e9), (180.0, 1e9)]):
    r = D.solve_point(5000 + i, L, nx, f, vg); rows.append(r)
    print(f"L={L} n_X={nx:.0e} {r['method']} maxiter={r['maxiter']} nfev={r['nfev']} R={r['R']:.4e} {r['seconds']:.1f}s", flush=True)
os.makedirs("results/da_smoke_boost", exist_ok=True)
json.dump(dict(rows=rows, host=os.uname().nodename), open("results/da_smoke_boost/smoke.json", "w"), indent=1)
