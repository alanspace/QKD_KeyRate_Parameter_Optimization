"""Independent solver cross-check of a reference file.

Independent of the qkd_opt optimiser in everything except the physics:
  * sampling:          pseudo-random uniform points (numpy PCG64, seed 12345) in the original
                       parameter box, rejection-sampled for feasibility -- not Sobol;
  * parameterisation:  original p = [mu1, mu2, P_mu1, P_mu2, q_x] with explicit inequality
                       constraints -- not the logit/stick-breaking map;
  * local solver:      scipy SLSQP (gradients from jax.grad of the same physics) -- not Newton.
For every reference point with R_ref > 0: draw 2048 feasible random points, refine the best 16
(diverse) with SLSQP, keep the best feasible result.
Reports d = (R_ref - R_xcheck)/R_ref. Any point with d < -1e-9 (cross-check better than the
reference) is a REFERENCE WEAKNESS and is listed.

Usage: python Optimization/reference/crosscheck_scipy.py --preset lim|lim_literal|legacy [--tail]
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from src.qkd_opt import get_preset, check_x64  # noqa: E402
from src.qkd_opt.objective import evaluate  # noqa: E402
from src.qkd_opt.provenance import stamp, to_jsonable  # noqa: E402

BOX_LO = np.array([4e-4, 2e-4, 1e-12, 1e-12, 1e-12])
BOX_HI = np.array([0.9, 0.5, 0.999, 0.999, 0.999])
N_RANDOM, N_REFINE, MIN_GAP = 2048, 16, 1e-9
POINT_BUDGET_S = 300.0   # per-point wall budget; remaining SLSQP starts are skipped (counted as timeout)


def feasible(P, mu3):
    return (P[:, 0] > P[:, 1] + mu3 + MIN_GAP) & (P[:, 1] > mu3 + MIN_GAP) & (P[:, 2] + P[:, 3] < 1 - MIN_GAP)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True)
    ap.add_argument("--tail", action="store_true")
    a = ap.parse_args()
    pre = get_preset(a.preset)
    check_x64()
    mu3 = pre.consts.mu3
    rpath = os.path.join(ROOT, "Optimization", "reference", f"reference_{pre.name}{'_tail' if a.tail else ''}.json")
    ref = json.load(open(rpath))
    cfg = dict(script="crosscheck_scipy.py", preset=pre.name, reference=os.path.relpath(rpath, ROOT),
               reference_config_hash=ref["stamp"]["config_hash"], n_random=N_RANDOM, n_refine=N_REFINE,
               rng="numpy PCG64 seed 12345", local="scipy SLSQP ftol 1e-14 maxiter 1000")
    st = stamp(cfg)
    batchR = jax.jit(jax.vmap(lambda p, L, n: evaluate(p, L, n, pre.consts, pre.flags)["R"], in_axes=(0, None, None)))
    vg = jax.jit(jax.value_and_grad(lambda p, L, n, s: -evaluate(p, L, n, pre.consts, pre.flags)["R"] / s))
    cons = [dict(type="ineq", fun=lambda p: p[0] - p[1] - mu3 - MIN_GAP, jac=lambda p: np.array([1., -1, 0, 0, 0])),
            dict(type="ineq", fun=lambda p: p[1] - mu3 - MIN_GAP, jac=lambda p: np.array([0., 1, 0, 0, 0])),
            dict(type="ineq", fun=lambda p: 1 - MIN_GAP - p[2] - p[3], jac=lambda p: np.array([0., 0, -1, -1, 0]))]
    rng = np.random.default_rng(12345)
    rows = []
    t0 = time.perf_counter()
    for k, p in enumerate(ref["points"]):
        if not (p["R"] > 0):
            continue
        L, n = p["L"], p["n_X"]
        P = np.empty((0, 5))
        while len(P) < N_RANDOM:
            c = rng.uniform(BOX_LO, BOX_HI, size=(4 * N_RANDOM, 5))
            P = np.concatenate([P, c[feasible(c, mu3)]])[:N_RANDOM]
        R0 = np.asarray(batchR(jnp.asarray(P), L, n))
        R0 = np.where(np.isfinite(R0), R0, -np.inf)
        order = np.argsort(-R0)
        starts = []
        for i in order:
            if all(np.max(np.abs(P[i] - P[j]) / (BOX_HI - BOX_LO)) > 0.05 for j in starts):
                starts.append(i)
            if len(starts) == N_REFINE:
                break
        s = max(abs(R0[order[0]]), 1e-300)
        best, bestp = -np.inf, None
        n_fail = n_infeas = n_nonfinite = 0
        timeout = False
        tp = time.perf_counter()
        for i in starts:
            if time.perf_counter() - tp > POINT_BUDGET_S:
                timeout = True
                break
            fg = lambda x: tuple(np.asarray(v, float) for v in vg(jnp.asarray(x), L, n, s))
            res = minimize(fg, P[i], jac=True, method="SLSQP", bounds=list(zip(BOX_LO, BOX_HI)),
                           constraints=cons, options=dict(ftol=1e-14, maxiter=1000))
            n_fail += int(not res.success)
            x = np.clip(res.x, BOX_LO, BOX_HI)
            if not feasible(x[None], mu3)[0]:
                n_infeas += 1
                continue
            Rx = float(evaluate(jnp.asarray(x), L, n, pre.consts, pre.flags)["R"])
            n_nonfinite += int(not np.isfinite(Rx))
            if np.isfinite(Rx) and Rx > best:
                best, bestp = Rx, x
        d = (p["R"] - best) / p["R"]
        rows.append(dict(L=L, n_X=n, R_ref=p["R"], R_xcheck=best, d=d, params_xcheck=bestp, params_ref=p["params"],
                         cliff_edge=bool(p.get("cliff_edge", False)), slsqp_not_success=n_fail,
                         infeasible_results=n_infeas, nonfinite=n_nonfinite, timeout=timeout,
                         seconds=time.perf_counter() - tp))
        if k % 20 == 0:
            print(f"[xcheck] {k}/{len(ref['points'])} L={L:.2f} n_X={n:.0e} d={d:.2e}", flush=True)
    d = np.array([r["d"] for r in rows])
    weak = [r for r in rows if r["d"] < -1e-9]
    summary = dict(n=len(rows), median_d=float(np.median(d)), max_d=float(np.max(d)), min_d=float(np.min(d)),
                   pct_agree_1e6=float(100 * np.mean(np.abs(d) <= 1e-6)),
                   n_xcheck_worse_gt_1e6=int(np.sum(d > 1e-6)), n_reference_weakness=len(weak),
                   slsqp_not_success_total=int(sum(r["slsqp_not_success"] for r in rows)),
                   slsqp_runs_total=int(len(rows) * N_REFINE),
                   infeasible_results_total=int(sum(r["infeasible_results"] for r in rows)),
                   nonfinite_total=int(sum(r["nonfinite"] for r in rows)),
                   n_points_timeout=int(sum(r["timeout"] for r in rows)),
                   n_points_no_feasible_result=int(sum(not np.isfinite(r["R_xcheck"]) for r in rows)),
                   wall_time_s=time.perf_counter() - t0)
    out_dir = os.path.join(ROOT, "results", f"crosscheck_{pre.name}{'_tail' if a.tail else ''}_{st['git_commit']}_{st['config_hash']}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "crosscheck.json"), "w") as fh:
        json.dump(to_jsonable(dict(stamp=st, config=cfg, summary=summary, points=rows)), fh, indent=1)
    md = [f"# Independent solver cross-check — preset `{pre.name}`{' (tail reference)' if a.tail else ''}", "",
          f"`Optimization/reference/crosscheck_scipy.py`; git `{st['git_commit']}`, config `{st['config_hash']}`; "
          f"reference `{cfg['reference']}` (config `{cfg['reference_config_hash']}`).", "",
          "d = (R_ref − R_xcheck)/R_ref; d > 0: cross-check worse than reference; d < −1e-9: REFERENCE WEAKNESS.", "",
          f"- points (R_ref > 0): {summary['n']}",
          f"- |d| ≤ 1e-6: {summary['pct_agree_1e6']:.1f} %",
          f"- median d: {summary['median_d']:.2e}; max d: {summary['max_d']:.2e}; min d: {summary['min_d']:.2e}",
          f"- cross-check worse than reference by > 1e-6: {summary['n_xcheck_worse_gt_1e6']} points",
          f"- **reference weaknesses (cross-check better by > 1e-9): {summary['n_reference_weakness']}**",
          f"- SLSQP runs not reporting success: {summary['slsqp_not_success_total']} of {summary['slsqp_runs_total']}; "
          f"infeasible results: {summary['infeasible_results_total']}; non-finite: {summary['nonfinite_total']}; "
          f"points hitting the {POINT_BUDGET_S:.0f} s budget: {summary['n_points_timeout']}; "
          f"points with no feasible result: {summary['n_points_no_feasible_result']}"]
    for r in weak:
        md.append(f"  - L={r['L']:.3f} km, n_X={r['n_X']:.0e}: R_ref={r['R_ref']:.10e}, R_xcheck={r['R_xcheck']:.10e}, "
                  f"d={r['d']:.2e}{' (cliff_edge reference point)' if r['cliff_edge'] else ''}")
    with open(os.path.join(out_dir, "CROSSCHECK.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
