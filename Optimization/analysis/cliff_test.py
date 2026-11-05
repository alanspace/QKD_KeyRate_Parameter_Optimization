"""Task C: the eps_sec = kappa*l discontinuity near the horizon (preset lim, default gamma).

Grids: n_X = 1e4, L = 128 .. 131 km step 0.25; n_X = 1e9, L = 182 .. 184.5 km step 0.25
(184.6 is not on the 0.25 grid; the last point is 184.5).

Method 1 (current default): maximise R(p) where eps_sec(p) is the greatest self-consistent fixed point
    of eps = kappa*max(l(p, eps), 1) for every p. Cold solve (2^15 Sobol, top-32, seed 2) plus the
    previous grid point's optimum as an extra start.
Method 2 (outer loop): eps_0 = kappa*n_X (>= any fixed point); repeat
    p_k = argmax_p R(p | eps fixed = eps_k)   (smooth fixed-eps problem; 2^14 Sobol, top-16, seed 2,
                                               plus p_{k-1} as extra start)
    eps_{k+1} = kappa * max(l(p_k | eps_k), 1)
    until |log eps_{k+1} - log eps_k| < 1e-10 (max 80 iterations; non-convergence is reported).
    The value map eps -> kappa*max(max_p l(p|eps),1) is increasing, so from eps_0 this decreases to its
    greatest fixed point.
Horizon per method: last grid point with l > 0, then 20-step bisection between it and the next grid
point using the same method. Reports whether the two horizons differ by more than 0.5 km.
Nothing here changes the default objective.

Usage: python Optimization/analysis/cliff_test.py
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from src.qkd_opt import LIM, check_x64, with_flags  # noqa: E402
from src.qkd_opt.optimize import optimize_batch  # noqa: E402
from src.qkd_opt.provenance import stamp, to_jsonable  # noqa: E402

GRIDS = {1e4: np.arange(128.0, 131.0 + 1e-9, 0.25), 1e9: np.arange(182.0, 184.6 + 1e-9, 0.25)}
FIXED = with_flags(LIM, "lim[eps fixed]", eps_mode="fixed")


def method1(L, nx, prev):
    r = optimize_batch([L], [nx], LIM, n_sobol=2**15, top_k=32, seed=2,
                       extra_starts=None if prev is None else prev[None, None, :])
    return dict(R=float(r["R"][0]), l=float(r["l"][0]), eps=float(r["eps_sec"][0]), params=r["params"][0],
                status=r["status"][0], grad=float(r["grad_norm_theta"][0]))


def method2(L, nx, prev):
    kappa = LIM.consts.kappa
    eps = kappa * nx
    p = prev
    hist = []
    conv = False
    for k in range(80):
        pre = FIXED._replace(consts=LIM.consts._replace(eps_sec=eps))
        r = optimize_batch([L], [nx], pre, n_sobol=2**14, top_k=16, seed=2,
                           extra_starts=None if p is None else p[None, None, :])
        p, l = r["params"][0], float(r["l"][0])
        new = kappa * max(l, 1.0)
        hist.append(dict(eps=eps, l=l))
        if abs(np.log(new) - np.log(eps)) < 1e-10:
            eps, conv = new, True
            break
        eps = new
    pre = FIXED._replace(consts=LIM.consts._replace(eps_sec=eps))
    r = optimize_batch([L], [nx], pre, n_sobol=2**14, top_k=16, seed=2, extra_starts=p[None, None, :])
    return dict(R=float(r["R"][0]), l=float(r["l"][0]), eps=eps, params=r["params"][0], status=r["status"][0],
                grad=float(r["grad_norm_theta"][0]), outer_iterations=len(hist), converged=conv)


def horizon(method, grid, rows, nx):
    pos = [i for i, r in enumerate(rows) if r["l"] > 0]
    if not pos:
        return dict(grid=None, bisected=None, note="l <= 0 on the whole grid")
    k = pos[-1]
    if k == len(grid) - 1:
        return dict(grid=float(grid[k]), bisected=None, note="l > 0 up to the grid end")
    lo, hi, p = float(grid[k]), float(grid[k + 1]), rows[k]["params"]
    for _ in range(20):
        mid = 0.5 * (lo + hi)
        r = method(mid, nx, p)
        if r["l"] > 0:
            lo, p = mid, r["params"]
        else:
            hi = mid
    return dict(grid=float(grid[k]), bisected=lo, bracket=[lo, hi])


def main():
    check_x64()
    cfg = dict(script="cliff_test.py", preset="lim", grids={str(k): v.tolist() for k, v in GRIDS.items()},
               method1="cold 2^15/top-32 seed 2 + previous optimum", method2="outer fixed-eps loop, 2^14/top-16")
    st = stamp(cfg)
    t0 = time.perf_counter()
    out = dict(stamp=st, config=cfg, n_X={})
    for nx, grid in GRIDS.items():
        rows1, rows2, p1, p2 = [], [], None, None
        for L in grid:
            a = method1(L, nx, p1)
            b = method2(L, nx, p2)
            p1, p2 = a["params"], b["params"]
            rows1.append(a)
            rows2.append(b)
            print(f"[cliff] n_X={nx:.0e} L={L:.2f}: m1 l={a['l']:+.4e} R={a['R']:+.4e} | "
                  f"m2 l={b['l']:+.4e} R={b['R']:+.4e} it={b['outer_iterations']} conv={b['converged']}", flush=True)
        h1 = horizon(method1, grid, rows1, nx)
        h2 = horizon(method2, grid, rows2, nx)
        d = None if (h1.get("bisected") is None or h2.get("bisected") is None) else abs(h1["bisected"] - h2["bisected"])
        out["n_X"][f"{nx:.0e}"] = dict(L=grid, method1=rows1, method2=rows2, horizon_method1=h1, horizon_method2=h2,
                                       horizon_abs_diff_km=d,
                                       differs_more_than_0p5km=None if d is None else bool(d > 0.5))
    out["wall_time_s"] = time.perf_counter() - t0
    d = os.path.join(ROOT, "results", f"cliff_test_{st['git_commit']}_{st['config_hash']}")
    os.makedirs(d, exist_ok=True)
    json.dump(to_jsonable(out), open(os.path.join(d, "cliff_test.json"), "w"), indent=1)
    md = [f"cliff test (preset lim), git {st['git_commit']}, config {st['config_hash']}, wall {out['wall_time_s']:.0f} s", ""]
    for k, v in out["n_X"].items():
        md.append(f"n_X = {k}")
        md.append("  L_km   method1_l   method1_R   method2_l   method2_R   m2_outer_iter m2_converged")
        for L, a, b in zip(v["L"], v["method1"], v["method2"]):
            md.append(f"  {L:6.2f}  {a['l']:+.4e}  {a['R']:+.4e}  {b['l']:+.4e}  {b['R']:+.4e}  "
                      f"{b['outer_iterations']:3d} {b['converged']}")
        md.append(f"  horizon method1: {v['horizon_method1']}")
        md.append(f"  horizon method2: {v['horizon_method2']}")
        md.append(f"  |difference| = {v['horizon_abs_diff_km']} km; differs by more than 0.5 km: {v['differs_more_than_0p5km']}")
        md.append("")
    open(os.path.join(d, "CLIFF_TEST.txt"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
