"""Measured (never modelled) timings on THIS machine, for one physics preset (--preset).

1. legacy DA per point: the DA algorithm and settings of the archived BB84_PureDualAnnealing.py.
   For --preset legacy this is the exact archived call on src.qkd's scalar_objective; for lim /
   lim_literal the same dual_annealing settings are applied to -R of that preset (infeasible points,
   which the box bounds allow, return 1e9 like the archived NaN guard). The archived run
   (scipy dual_annealing on jit(src.qkd.model.scalar_objective), its stepped maxiter schedule,
   initial_temp=5230, visit=2.62, accept=-5, no_local_search=True, x0 = its default guess) at sample
   fiber lengths > 100 km, plus its L-BFGS-B branch (maxiter 500, ftol 1e-12) at L <= 100 km.
   JIT compile excluded (one warm-up call). Seeded (seed=0) for reproducibility; the original was
   unseeded. Single process.
2. warm-started JAX solver, one point (batch 1): the jitted local kernel local_batch with
   device-resident inputs, timed with block_until_ready  -> "steady-state (device-resident)";
   and the Python API warm_solve() from numpy in to numpy out -> "end-to-end".
3. lookup table (linear interpolation in L and log10 n_X of a table of optimal params): jitted
   kernel on device -> steady-state; numpy in/out -> end-to-end.
4. NN surrogate: NOT measured -- torch/scikit-learn are not installed in this environment.

Reports median and p90 over repetitions. Usage: python Optimization/benchmark/timing_benchmark.py --preset lim|lim_literal|legacy
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from scipy.optimize import dual_annealing, minimize  # noqa: E402

from src.qkd_opt import get_preset, check_x64, reparam  # noqa: E402
from src.qkd_opt.objective import evaluate  # noqa: E402
from src.qkd_opt.optimize import local_batch, warm_solve, optimize_batch, eval_starts  # noqa: E402
from src.qkd_opt.provenance import stamp, to_jsonable  # noqa: E402
from src.qkd.model import scalar_objective, objective_val_and_grad  # noqa: E402

CONST = (0.2, 0.1, 6e-7, 1e-10, 1e-15, 1.16, 0.01, 0, 1)   # legacy script constants
BOUNDS = [(4e-4, 0.9), (2e-4, 0.5), (1e-12, 0.999), (1e-12, 0.999), (1e-12, 0.999)]
X0 = np.array([0.52, 0.40, 0.18, 0.785, 0.88])


def q(ts):
    ts = np.asarray(ts)
    return dict(median=float(np.median(ts)), p90=float(np.quantile(ts, 0.9)), n=int(ts.size))


def _objectives(pre):
    """(value fn, value_and_grad fn) of the minimisation objective for the DA / L-BFGS-B timings."""
    if pre.name == "legacy":
        f = jax.jit(scalar_objective)
        return (lambda p, L, nx: f(jnp.asarray(p), L, nx, *CONST),
                lambda p, L, nx: objective_val_and_grad(p, L, nx, *CONST))

    def negR(p, L, nx):
        p = jnp.asarray(p)
        feas = (p[0] > p[1] + pre.consts.mu3) & (p[1] > pre.consts.mu3) & (p[2] + p[3] < 1.0)
        return jnp.where(feas, -evaluate(p, L, nx, pre.consts, pre.flags)["R"], 1e9)
    return jax.jit(negR), jax.jit(jax.value_and_grad(negR))


def legacy_da(points, pre, reps=1):
    f, _ = _objectives(pre)
    jax.block_until_ready(f(jnp.asarray(X0), 120.0, 1e9))  # compile
    out = []
    for L, nx in points:
        L, nx = float(L), float(nx)
        it = 5000 if L <= 120 else 7500 if L <= 140 else 10000 if L <= 160 else 15000
        nfev = [0]

        def obj(p):
            nfev[0] += 1
            v = float(f(jnp.asarray(p), L, nx))
            return v if np.isfinite(v) else 1e9
        t = time.perf_counter()
        res = dual_annealing(obj, BOUNDS, maxiter=it, initial_temp=5230.0, visit=2.62, accept=-5.0, x0=X0,
                             no_local_search=True, seed=0)
        dt = time.perf_counter() - t
        out.append(dict(L=L, n_X=nx, maxiter=it, seconds=dt, nfev=nfev[0], R=-float(res.fun),
                        us_per_eval=1e6 * dt / nfev[0]))
        print(f"[timing] legacy DA L={L} n_X={nx:.0e}: {dt:.1f}s nfev={nfev[0]}", flush=True)
    return out


def legacy_lbfgsb(points, pre):
    _, vg = _objectives(pre)
    vg(jnp.asarray(X0), 50.0, 1e9)
    out = []
    for L, nx in points:
        L, nx = float(L), float(nx)   # int L would trigger a recompile (weak-type int)
        def fg(p):
            v, g = vg(jnp.asarray(p), L, nx)
            if not np.isfinite(v):
                return 1e9, np.zeros_like(p)
            return float(v), np.clip(np.array(g), -1e6, 1e6)
        t = time.perf_counter()
        res = minimize(fg, X0, method="L-BFGS-B", jac=True, bounds=BOUNDS, options={"maxiter": 500, "ftol": 1e-12})
        out.append(dict(L=L, n_X=nx, seconds=time.perf_counter() - t, nfev=int(res.nfev), R=-float(res.fun)))
    return out


def jax_warm(pre, reps=200):
    """Warm start from the optimum at L-0.5 km, solve at L (one point)."""
    L0, L1, nx = 99.5, 100.0, 1e9
    base = optimize_batch([L0], [nx], pre, n_sobol=4096, top_k=8)
    th0 = base["theta"]                     # [1,5]
    R0, l0 = eval_starts(jnp.asarray(th0)[:, None, :], jnp.asarray([L1]), jnp.asarray([nx]), pre.consts, pre.flags)
    scale = jnp.abs(R0[:, 0])
    args = (jnp.asarray(th0)[:, None, :], jnp.asarray([L1]), jnp.asarray([nx]), jnp.asarray([0]), scale,
            pre.consts, pre.flags)
    jax.block_until_ready(local_batch(*args))  # compile
    dev = []
    for _ in range(reps):
        t = time.perf_counter_ns(); jax.block_until_ready(local_batch(*args)); dev.append(time.perf_counter_ns() - t)
    th_np = np.asarray(th0)
    warm_solve(th_np, [L1], [nx], pre)  # compile host path
    e2e = []
    for _ in range(max(reps // 4, 20)):
        t = time.perf_counter_ns(); w = warm_solve(th_np, [L1], [nx], pre); e2e.append(time.perf_counter_ns() - t)
    iters = int(np.asarray(local_batch(*args)[2])[0, 0])
    return dict(steady_device_us=q(np.array(dev) / 1e3), end_to_end_us=q(np.array(e2e) / 1e3),
                newton_iterations=iters, point=dict(L_from=L0, L=L1, n_X=nx))


def lookup(pre, reps=2000):
    """Table of optimal params on (L grid 0.5 km) x (log10 n_X = 4..9) from a coarse sweep."""
    from src.qkd_opt.optimize import optimize_sweep
    grid = np.linspace(0, 200, 81)
    nxs = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
    sw = optimize_sweep(grid, nxs, pre, global_every=10, refine_horizon=False)
    T = jnp.asarray(sw["params"])            # [6, nL, 5]
    gL = jnp.asarray(grid)
    gN = jnp.arange(4.0, 10.0)

    @jax.jit
    def interp(L, lognx):
        i = jnp.clip(jnp.searchsorted(gN, lognx) - 1, 0, gN.size - 2)
        w = (lognx - gN[i]) / (gN[i + 1] - gN[i])
        rowa = jax.vmap(lambda c: jnp.interp(L, gL, T[i, :, c]))(jnp.arange(5))
        rowb = jax.vmap(lambda c: jnp.interp(L, gL, T[i + 1, :, c]))(jnp.arange(5))
        return (1 - w) * rowa + w * rowb

    a = (jnp.asarray(100.0), jnp.asarray(9.0))
    jax.block_until_ready(interp(*a))
    dev = []
    for _ in range(reps):
        t = time.perf_counter_ns(); jax.block_until_ready(interp(*a)); dev.append(time.perf_counter_ns() - t)
    e2e = []
    for _ in range(reps):
        t = time.perf_counter_ns(); np.asarray(interp(float(100.0), float(9.0))); e2e.append(time.perf_counter_ns() - t)
    return dict(steady_device_us=q(np.array(dev) / 1e3), end_to_end_us=q(np.array(e2e) / 1e3),
                table_shape=list(T.shape))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True)
    pre = get_preset(ap.parse_args().preset)
    from src.qkd_opt.provenance import power_state
    pw = power_state()
    if pw["source"] != "AC":
        sys.exit(f"timing_benchmark: refusing to run, power source is {pw['source']!r} "
                 f"({pw['battery_percent']}% {pw['battery_state']}); connect AC power")
    backend = check_x64()
    cfg = dict(script="timing_benchmark.py", preset=pre.name, da_points=[[110, 1e9], [150, 1e9], [190, 1e9],
                                                                         [130, 1e4], [145, 1e4]],
               lbfgsb_points=[[0, 1e9], [50, 1e9], [100, 1e9], [50, 1e4]])
    st = stamp(cfg)
    res = dict(stamp=st, config=cfg, backend=backend)
    res["legacy_lbfgsb_branch"] = legacy_lbfgsb(cfg["lbfgsb_points"], pre)
    res["jax_warm_batch1"] = jax_warm(pre)
    print("[timing] jax warm:", res["jax_warm_batch1"], flush=True)
    res["lookup_table_batch1"] = lookup(pre)
    print("[timing] lookup:", res["lookup_table_batch1"], flush=True)
    res["legacy_DA"] = legacy_da(cfg["da_points"], pre)
    res["nn_surrogate"] = "not measured: torch / scikit-learn not installed in this environment"
    out = os.path.join(ROOT, "results", f"timing_{pre.name}_{st['git_commit']}_{st['config_hash']}")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "timing.json"), "w") as fh:
        json.dump(to_jsonable(res), fh, indent=1)
    da = res["legacy_DA"]
    lines = ["# Timing benchmark (measured on this machine)", "",
             f"git `{st['git_commit']}`, config `{st['config_hash']}`, {st['cpu']} ({st['n_cpu']} logical CPUs), "
             f"JAX {st['jax']} `{st['jax_backend']}`, float64. Preset `{pre.name}`.", "",
             "| method | quantity | median | p90 | notes |", "|---|---|---|---|---|"]
    w, lt = res["jax_warm_batch1"], res["lookup_table_batch1"]
    lines.append(f"| JAX warm-started solver | batch-1 steady-state, device-resident (µs) | "
                 f"{w['steady_device_us']['median']:.0f} | {w['steady_device_us']['p90']:.0f} | "
                 f"{w['newton_iterations']} Newton iterations, warm start from L−0.5 km |")
    lines.append(f"| JAX warm-started solver | end-to-end Python API (µs) | {w['end_to_end_us']['median']:.0f} | "
                 f"{w['end_to_end_us']['p90']:.0f} | numpy in → numpy out |")
    lines.append(f"| lookup-table interpolation | batch-1 steady-state, device-resident (µs) | "
                 f"{lt['steady_device_us']['median']:.1f} | {lt['steady_device_us']['p90']:.1f} | table {lt['table_shape']} |")
    lines.append(f"| lookup-table interpolation | end-to-end (µs) | {lt['end_to_end_us']['median']:.1f} | "
                 f"{lt['end_to_end_us']['p90']:.1f} | |")
    for d in da:
        lines.append(f"| legacy DA ({'archived call' if pre.name == 'legacy' else 'archived settings, this preset'}) | one point (s) | {d['seconds']:.1f} | – | L={d['L']} km, "
                     f"n_X={d['n_X']:.0e}, maxiter={d['maxiter']}, nfev={d['nfev']}, {d['us_per_eval']:.0f} µs/eval |")
    for d in res["legacy_lbfgsb_branch"]:
        lines.append(f"| legacy L-BFGS-B branch (L ≤ 100 km) | one point (s) | {d['seconds']:.3f} | – | "
                     f"L={d['L']} km, n_X={d['n_X']:.0e}, nfev={d['nfev']} |")
    lines.append("| NN surrogate | – | – | – | not measured (torch not installed) |")
    with open(os.path.join(out, "TIMING.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
