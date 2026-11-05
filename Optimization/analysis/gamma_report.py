"""Task E: printed vs exact (Fung-Ma-Chau) vs max gamma in Eq. (5), preset lim otherwise unchanged.

Per gamma_mode (printed = preset lim, exact = preset lim_tight, max):
  * horizon (largest L with l > 0) for every n_X: bracket around the lim reference horizon (+/- 25 km,
    widened in 25 km steps until l(lo) > 0 >= l(hi)), then 22-step bisection; every step is a cold solve
    (2^14 Sobol, top-16, seed 4) plus the lower bracket's optimum as an extra start;
  * R(0, 1e9) and R(100 km, 1e9)/R(100 km, 1e7) (cold 2^15, top-16);
  * paper T3 = HORIZON at n_X = 1e4 (paper ~135 km); secondary: L where R falls to 1e-7 at n_X = 1e4
    (1 km sweep 90..150 km + 25-step bisection).
Operating points for the gamma figure: at every lim reference point with R > 0, b = v_Z1/s_Z1,
c = s_Z1, d = s_X1, eps_sec, printed gamma and exact theta (same eps/21 mapping).

Usage: python Optimization/analysis/gamma_report.py
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from src.qkd_opt import LIM, LIM_TIGHT, check_x64, with_flags, evaluate  # noqa: E402
from src.qkd_opt.optimize import optimize_batch, optimize_sweep, warm_solve, eval_starts  # noqa: E402
from src.qkd_opt.physics_lim import _gamma, theta_exact  # noqa: E402
from src.qkd_opt.provenance import stamp, to_jsonable  # noqa: E402

NX = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
MODES = {"printed": LIM, "exact": LIM_TIGHT, "max": with_flags(LIM, "lim[gamma=max]", gamma_mode="max")}


def l_at(L, nx, pre, extra=None):
    r = optimize_batch([L], [nx], pre, n_sobol=2**14, top_k=16, seed=4,
                       extra_starts=None if extra is None else extra[None, None, :])
    return float(r["l"][0]), r["params"][0]


def horizon(nx, pre, guess):
    lo, hi = max(guess - 25.0, 0.0), guess + 25.0
    llo, plo = l_at(lo, nx, pre)
    while llo <= 0 and lo > 0:
        lo = max(lo - 25.0, 0.0)
        llo, plo = l_at(lo, nx, pre)
    if llo <= 0:
        return dict(horizon_km=None, note="l <= 0 at L = 0")
    lhi, _ = l_at(hi, nx, pre, plo)
    while lhi > 0 and hi < 400:
        hi += 25.0
        lhi, _ = l_at(hi, nx, pre, plo)
    for _ in range(22):
        mid = 0.5 * (lo + hi)
        lm, pm = l_at(mid, nx, pre, plo)
        if lm > 0:
            lo, plo = mid, pm
        else:
            hi = mid
    return dict(horizon_km=lo, bracket_km=[lo, hi])


def crossing_1e7(pre):
    Lg = np.arange(90.0, 151.0, 1.0)
    sw = optimize_sweep(Lg, [1e4], pre, global_every=5, refine_horizon=False)
    above = np.where(sw["R"][0] >= 1e-7)[0]
    if len(above) == 0:
        return None
    k = above[-1]
    if k == len(Lg) - 1:
        return None
    lo, hi, th = Lg[k], Lg[k + 1], sw["theta"][0, k]
    for _ in range(25):
        mid = 0.5 * (lo + hi)
        w = warm_solve(th[None], [mid], [1e4], pre)
        R = float(np.asarray(eval_starts(jnp.asarray(w["theta"])[:, None, :], jnp.asarray([mid]), jnp.asarray([1e4]),
                                         pre.consts, pre.flags)[0])[0, 0])
        if R >= 1e-7:
            lo, th = mid, w["theta"][0]
        else:
            hi = mid
    return float(lo)


def main():
    check_x64()
    ref = json.load(open(os.path.join(ROOT, "Optimization", "reference", "reference_lim.json")))
    hz = {h["n_X"]: h.get("horizon_km") for h in ref["horizons"]}
    cfg = dict(script="gamma_report.py", modes=list(MODES), guess_horizons=hz,
               reference_config_hash=ref["stamp"]["config_hash"])
    st = stamp(cfg)
    t0 = time.perf_counter()
    out = dict(stamp=st, config=cfg, modes={})
    for m, pre in MODES.items():
        t = time.perf_counter()
        hs = {f"{nx:.0e}": horizon(nx, pre, hz[nx]) for nx in NX}
        r = optimize_batch([0.0, 100.0, 100.0], [1e9, 1e9, 1e7], pre, n_sobol=2**15, top_k=16, seed=4)
        out["modes"][m] = dict(horizons=hs, R0_1e9=float(r["R"][0]), ratio_100km=float(r["R"][1] / r["R"][2]),
                               T3_horizon_nX1e4=hs["1e+04"]["horizon_km"], T3b_L_at_R_1e7_nX1e4=crossing_1e7(pre),
                               seconds=time.perf_counter() - t)
        print(f"[gamma] {m}: {json.dumps(to_jsonable(out['modes'][m]))}", flush=True)
    # operating points
    ops = []
    for p in ref["points"]:
        if not p["R"] > 0:
            continue
        o = evaluate(jnp.asarray(p["params"]), p["L"], p["n_X"], LIM.consts, LIM.flags)
        b = float(o["vZ1"] / o["sZ1"]); c = float(o["sZ1"]); d = float(o["sX1"]); eps = float(o["eps_sec"])
        if not (0 < b < 0.5 and c > 0 and d > 0):
            continue
        gp = float(_gamma(eps, b, c, d))
        te = float(theta_exact(jnp.asarray(eps / 21.0), jnp.asarray(b), jnp.asarray(c), jnp.asarray(d)))
        ops.append(dict(L=p["L"], n_X=p["n_X"], b=b, c=c, d=d, eps=eps, gamma_printed=gp, theta_exact=te,
                        ratio=gp / te if te > 0 else None))
    out["operating_points"] = ops
    out["wall_time_s"] = time.perf_counter() - t0
    dd = os.path.join(ROOT, "results", f"gamma_report_{st['git_commit']}_{st['config_hash']}")
    os.makedirs(dd, exist_ok=True)
    json.dump(to_jsonable(out), open(os.path.join(dd, "gamma_report.json"), "w"), indent=1)
    md = [f"gamma report (preset lim; gamma printed / exact / max), git {st['git_commit']}, config {st['config_hash']}, "
          f"wall {out['wall_time_s']:.0f} s", ""]
    for m, v in out["modes"].items():
        md.append(f"gamma_mode={m}: R(0,1e9)={v['R0_1e9']:.6e}; R(100,1e9)/R(100,1e7)={v['ratio_100km']:.4f}; "
                  f"T3 horizon n_X=1e4 = {v['T3_horizon_nX1e4']} km (paper ~135); T3b L at R=1e-7 = {v['T3b_L_at_R_1e7_nX1e4']} km")
        md.append("  horizons km: " + ", ".join(f"n_X={k}: {h['horizon_km']}" for k, h in v["horizons"].items()))
    rat = [o["ratio"] for o in ops if o["ratio"]]
    md += ["", f"operating points (lim reference, R>0): {len(ops)}; printed gamma / exact theta: "
               f"min {min(rat):.4f}, median {float(np.median(rat)):.4f}, max {max(rat):.4f}" if rat else "no operating points"]
    open(os.path.join(dd, "GAMMA_REPORT.txt"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
