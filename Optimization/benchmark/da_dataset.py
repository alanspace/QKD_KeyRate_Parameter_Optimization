"""Regenerate a dual-annealing (DA) labelled dataset with the archived DA pipeline, in chunks (HPC array jobs),
and score it against the verified optimizer at the same points.

Pipeline (identical to timing_benchmark.legacy_da / legacy_lbfgsb, i.e. archived BB84_PureDualAnnealing.py):
  L <= 100 km : L-BFGS-B from the archived default guess (maxiter 500, ftol 1e-12)
  L >  100 km : scipy dual_annealing, maxiter 5000/7500/10000/15000 by L, initial_temp 5230, visit 2.62,
                accept -5, no_local_search, x0 = archived default guess, seed = point index (reproducible)
Objective: -R of the chosen preset (infeasible -> 1e9, as the archived NaN guard); legacy uses src.qkd itself.
Default grid: the 2025 thesis grid, L = linspace(0, 200, 1000) x n_X = 1e4..1e9 (6,000 points).

  run    --chunk i --nchunks N   process every N-th point starting at i; writes results/da_dataset_<preset>/chunk_i.json
  merge                          collect chunks; solve every point with the verified optimizer (optimize_batch,
                                 2^12 Sobol, top-8, seed 7); write results/da_dataset_<preset>/DA_DATASET.md + .json
                                 (regret of DA vs optimizer; % within 1e-6 / 1e-3 / 1e-2; % no key; time per point)

Usage:  python Optimization/benchmark/da_dataset.py run --preset lim --chunk 0 --nchunks 60
        python Optimization/benchmark/da_dataset.py merge --preset lim
"""
import argparse
import glob
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "Optimization", "benchmark"))

import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)
from scipy.optimize import dual_annealing, minimize  # noqa: E402
from src.qkd_opt import get_preset, check_x64  # noqa: E402
from src.qkd_opt.provenance import stamp, to_jsonable  # noqa: E402
import timing_benchmark as TB  # noqa: E402  (archived settings: BOUNDS, X0, _objectives)

NX = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9]


def grid(n_L=1000):
    L = np.linspace(0.0, 200.0, n_L)
    return [(float(l_), float(nx)) for nx in NX for l_ in L]


def out_dir(preset):
    d = os.path.join(ROOT, "results", f"da_dataset_{preset}")
    os.makedirs(d, exist_ok=True)
    return d


def solve_point(i, L, nx, f, vg):
    t = time.perf_counter()
    nfev = [0]
    if L <= 100.0:
        def fg(p):
            nfev[0] += 1
            v, g = vg(jnp.asarray(p), L, nx)
            if not np.isfinite(v):
                return 1e9, np.zeros_like(p)
            return float(v), np.clip(np.array(g), -1e6, 1e6)
        res = minimize(fg, TB.X0, method="L-BFGS-B", jac=True, bounds=TB.BOUNDS,
                       options={"maxiter": 500, "ftol": 1e-12})
        method, it = "L-BFGS-B", 500
    else:
        it = 5000 if L <= 120 else 7500 if L <= 140 else 10000 if L <= 160 else 15000

        def obj(p):
            nfev[0] += 1
            v = float(f(jnp.asarray(p), L, nx))
            return v if np.isfinite(v) else 1e9
        res = dual_annealing(obj, TB.BOUNDS, maxiter=it, initial_temp=5230.0, visit=2.62, accept=-5.0, x0=TB.X0,
                             no_local_search=True, seed=int(i))
        method = "dual_annealing"
    return dict(index=int(i), L=L, n_X=nx, method=method, maxiter=it, params=[float(x) for x in res.x],
                R=-float(res.fun), nfev=nfev[0], seconds=time.perf_counter() - t)


def cmd_run(a):
    check_x64()
    pre = get_preset(a.preset)
    f, vg = TB._objectives(pre)
    jax.block_until_ready(f(jnp.asarray(TB.X0), 120.0, 1e9))
    vg(jnp.asarray(TB.X0), 50.0, 1e9)
    pts = grid(a.n_L)
    mine = [(i, p) for i, p in enumerate(pts) if i % a.nchunks == a.chunk]
    if a.limit:
        mine = mine[:a.limit]
    cfg = dict(script="Optimization/benchmark/da_dataset.py run", preset=pre.name, n_L=a.n_L, nX=NX,
               chunk=a.chunk, nchunks=a.nchunks, limit=a.limit)
    rows = []
    for i, (L, nx) in mine:
        r = solve_point(i, L, nx, f, vg)
        rows.append(r)
        print(f"[da] {i} L={L:.3f} n_X={nx:.0e} {r['method']} R={r['R']:.4e} {r['seconds']:.1f}s", flush=True)
    path = os.path.join(out_dir(pre.name), f"chunk_{a.chunk:04d}_of_{a.nchunks:04d}.json")
    with open(path, "w") as fh:
        json.dump(to_jsonable(dict(stamp=stamp(cfg), config=cfg, rows=rows)), fh)
    print(f"[da] wrote {os.path.relpath(path, ROOT)} ({len(rows)} points)", flush=True)


def cmd_merge(a):
    from src.qkd_opt.optimize import optimize_batch
    check_x64()
    pre = get_preset(a.preset)
    od = out_dir(pre.name)
    files = sorted(glob.glob(os.path.join(od, "chunk_*.json")))
    rows = [r for f in files for r in json.load(open(f))["rows"]]
    rows.sort(key=lambda r: r["index"])
    n_expected = len(grid(a.n_L))
    L = np.array([r["L"] for r in rows]); nX = np.array([r["n_X"] for r in rows]); R_da = np.array([r["R"] for r in rows])
    best = optimize_batch(L, nX, pre, n_sobol=4096, top_k=8, seed=7)
    R_opt = np.asarray(best["R"])
    pos = R_opt > 0
    reg = np.where(pos, (R_opt - R_da) / np.where(pos, R_opt, 1.0), np.nan)
    secs = np.array([r["seconds"] for r in rows]); meth = np.array([r["method"] for r in rows])

    def summ(m):
        x = reg[m & pos]
        return dict(n=int(x.size), median=float(np.median(x)) if x.size else None,
                    pct_le_1e6=float(100 * np.mean(x <= 1e-6)) if x.size else None,
                    pct_le_1e3=float(100 * np.mean(x <= 1e-3)) if x.size else None,
                    pct_le_1e2=float(100 * np.mean(x <= 1e-2)) if x.size else None,
                    pct_no_key=float(100 * np.mean(x >= 1)) if x.size else None)
    res = dict(stamp=stamp(dict(script="da_dataset.py merge", preset=pre.name, n_rows=len(rows))),
               n_rows=len(rows), n_expected=n_expected, complete=len(rows) == n_expected,
               all=summ(np.ones(len(rows), bool)),
               by_method={m: summ(meth == m) for m in np.unique(meth)},
               by_nX={f"{nx:.0e}": summ(nX == nx) for nx in NX},
               seconds_per_point=dict(lbfgsb_median=float(np.median(secs[meth == "L-BFGS-B"])) if (meth == "L-BFGS-B").any() else None,
                                      da_median=float(np.median(secs[meth == "dual_annealing"])) if (meth == "dual_annealing").any() else None,
                                      total_cpu_h=float(secs.sum() / 3600)),
               points=[dict(r, R_opt=float(ro), regret=float(g)) for r, ro, g in zip(rows, R_opt, reg)])
    with open(os.path.join(od, "da_dataset.json"), "w") as fh:
        json.dump(to_jsonable(res), fh)
    s = res["all"]
    md = [f"# DA dataset ({pre.name}): {len(rows)}/{n_expected} points", "",
          f"Regret of the archived DA pipeline vs the verified optimizer at the same points (R_opt > 0 only):",
          f"median {s['median']}, within 1e-6 {s['pct_le_1e6']} %, within 1e-3 {s['pct_le_1e3']} %, "
          f"within 1e-2 {s['pct_le_1e2']} %, no key {s['pct_no_key']} %.", "",
          f"Per method: {json.dumps(res['by_method'])}", "", f"Per n_X: {json.dumps(res['by_nX'])}", "",
          f"Seconds per point: {json.dumps(res['seconds_per_point'])}"]
    with open(os.path.join(od, "DA_DATASET.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "merge"])
    ap.add_argument("--preset", default="lim")
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--nchunks", type=int, default=1)
    ap.add_argument("--n-L", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=0, help="run: only the first N points of the chunk (smoke test)")
    a = ap.parse_args()
    {"run": cmd_run, "merge": cmd_merge}[a.cmd](a)


if __name__ == "__main__":
    main()
