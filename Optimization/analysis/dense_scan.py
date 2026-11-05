"""Task D: dense mid-range scan for branch switches (preset lim) + legacy comparison at n_X = 1e9.

For each n_X: L = 100 km .. (reference horizon + 1) km in 1 km steps, cold solves with the dense budget
(2^17 scrambled Sobol, top-64), seeds 2 and 3, best of the two kept.
A "jump" is a consecutive pair with max |dp / box width| > 0.05 (box = [0.9, 0.5, 1, 1, 1]).
For each jump (j, j+1), A = optimum at L_j, B = optimum at L_{j+1}:
  * cross-evaluation: warm-solve from A at L_{j+1} -> R_A(L_{j+1}); from B at L_j -> R_B(L_j)
    (each also reports whether it stayed on its branch: u-distance to its start < 0.02)
  * branch crossing (R continuous, kink) if both branches persist and R_B(L_j) <= R(L_j)(1+1e-9) and
    R_A(L_{j+1}) <= R(L_{j+1})(1+1e-9)
  * R continuity: d_j = dlogR(j,j+1) - mean(dlogR(j-1,j), dlogR(j+1,j+2)); "R discontinuous" if |d_j| > 0.05
Legacy comparison (n_X = 1e9): the same dense scan with preset legacy on L = 140 .. 160 km, plus the jump
locations of the stored legacy runs (re-read from their JSONs, 0.5 km grid) in 140 .. 160 km.

Usage: python Optimization/analysis/dense_scan.py
"""
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "Optimization", "benchmark"))

import numpy as np  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from src.qkd_opt import LIM, LEGACY, check_x64, reparam  # noqa: E402
from src.qkd_opt.optimize import optimize_batch, warm_solve, eval_starts  # noqa: E402
from src.qkd_opt.provenance import stamp, to_jsonable  # noqa: E402
import legacy_runs  # noqa: E402

BOX = np.array([reparam.MU1_MAX, reparam.MU2_MAX, 1.0, 1.0, 1.0])
NX = legacy_runs.NX
JUMP = 0.05


def dense(L, n, pre):
    best = None
    for s in (2, 3):
        r = optimize_batch(L, n, pre, n_sobol=2**17, top_k=64, seed=s)
        if best is None:
            best = r
        else:
            better = r["R"] > best["R"]
            for k in ("R", "l", "params", "theta", "grad_norm_theta"):
                best[k] = np.where(better[:, None] if best[k].ndim == 2 else better, r[k], best[k])
            best["status"] = np.where(better, r["status"], best["status"])
    return best


def R_at(theta, L, n, pre):
    R, _ = eval_starts(jnp.asarray(theta)[None, None, :], jnp.asarray([L]), jnp.asarray([n]), pre.consts, pre.flags)
    return float(np.asarray(R)[0, 0])


def analyse(Ls, n, res, pre):
    P, R, TH = res["params"], res["R"], res["theta"]
    d = np.abs(np.diff(P / BOX, axis=0)).max(axis=1)
    lr = np.log(np.where(R > 0, R, np.nan))
    dl = np.diff(lr)
    out = []
    for j in np.where(d > JUMP)[0]:
        wa = warm_solve(TH[j][None], [Ls[j + 1]], [n], pre)
        wb = warm_solve(TH[j + 1][None], [Ls[j]], [n], pre)
        RA1, RB0 = R_at(wa["theta"][0], Ls[j + 1], n, pre), R_at(wb["theta"][0], Ls[j], n, pre)
        ua, ub = (np.asarray(reparam.u_of_theta(jnp.asarray(t))) for t in (TH[j], TH[j + 1]))
        stayA = float(np.max(np.abs(np.asarray(reparam.u_of_theta(jnp.asarray(wa["theta"][0]))) - ua))) < 0.02
        stayB = float(np.max(np.abs(np.asarray(reparam.u_of_theta(jnp.asarray(wb["theta"][0]))) - ub))) < 0.02
        crossing = bool(stayA and stayB and RB0 <= R[j] * (1 + 1e-9) and RA1 <= R[j + 1] * (1 + 1e-9))
        nb = [dl[k] for k in (j - 1, j + 1) if 0 <= k < len(dl) and np.isfinite(dl[k])]
        dev = float(dl[j] - np.mean(nb)) if (nb and np.isfinite(dl[j])) else None
        out.append(dict(L_from=float(Ls[j]), L_to=float(Ls[j + 1]), max_norm_jump=float(d[j]),
                        R_from=float(R[j]), R_to=float(R[j + 1]),
                        R_branchA_at_L_to=RA1, R_branchB_at_L_from=RB0, branchA_persists=stayA,
                        branchB_persists=stayB, branch_crossing=crossing, dlogR_deviation=dev,
                        R_discontinuous=None if dev is None else bool(abs(dev) > 0.05),
                        params_from=P[j], params_to=P[j + 1]))
    return out


def stored_jumps(name, lo=140.0, hi=160.0):
    run = legacy_runs.load(name)
    Rr, _ = legacy_runs.reevaluate(run, LEGACY)
    i = NX.index(1e9)
    L, P = run["L"], run["params"][i]
    d = np.abs(np.diff(P / BOX, axis=0)).max(axis=1)
    sel = np.where((d > JUMP) & (L[:-1] >= lo) & (L[1:] <= hi) & (Rr[i][:-1] > 0) & (Rr[i][1:] > 0))[0]
    return [dict(L_from=float(L[j]), L_to=float(L[j + 1]), max_norm_jump=float(d[j])) for j in sel]


def main():
    check_x64()
    ref = json.load(open(os.path.join(ROOT, "Optimization", "reference", "reference_lim.json")))
    hz = {h["n_X"]: h.get("horizon_km") for h in ref["horizons"]}
    cfg = dict(script="dense_scan.py", preset="lim", n_sobol=2**17, top_k=64, seeds=[2, 3], step_km=1.0,
               reference_horizons=hz, reference_config_hash=ref["stamp"]["config_hash"], jump=JUMP)
    st = stamp(cfg)
    t0 = time.perf_counter()
    out = dict(stamp=st, config=cfg, lim={}, legacy_1e9={})
    for nx in NX:
        h = hz.get(nx)
        if h is None or h < 100:
            out["lim"][f"{nx:.0e}"] = dict(note=f"reference horizon {h} km < 100 km: nothing to scan")
            continue
        Ls = np.arange(100.0, np.floor(h + 1.0) + 1e-9, 1.0)
        t = time.perf_counter()
        res = dense(Ls, np.full_like(Ls, nx), LIM)
        jumps = analyse(Ls, nx, res, LIM)
        out["lim"][f"{nx:.0e}"] = dict(L=Ls, R=res["R"], params=res["params"], jumps=jumps,
                                       seconds=time.perf_counter() - t)
        print(f"[dense] lim n_X={nx:.0e}: {len(Ls)} points, {len(jumps)} jumps, {time.perf_counter() - t:.0f}s", flush=True)
    Ls = np.arange(140.0, 160.0 + 1e-9, 1.0)
    res = dense(Ls, np.full_like(Ls, 1e9), LEGACY)
    out["legacy_1e9"] = dict(L=Ls, R=res["R"], params=res["params"], jumps=analyse(Ls, 1e9, res, LEGACY),
                             stored_runs={k: stored_jumps(k) for k in ("legacy_DA", "legacy_Hessian", "legacy_Hessian_FoR")})
    out["wall_time_s"] = time.perf_counter() - t0
    d = os.path.join(ROOT, "results", f"dense_scan_{st['git_commit']}_{st['config_hash']}")
    os.makedirs(d, exist_ok=True)
    json.dump(to_jsonable(out), open(os.path.join(d, "dense_scan.json"), "w"), indent=1)
    md = [f"dense mid-range scan, git {st['git_commit']}, config {st['config_hash']}, wall {out['wall_time_s']:.0f} s",
          f"jump = consecutive 1 km pair with max |dp/box| > {JUMP}", ""]
    for k, v in out["lim"].items():
        if "note" in v:
            md.append(f"lim n_X={k}: {v['note']}")
            continue
        md.append(f"lim n_X={k}: L 100..{v['L'][-1]:.0f} km, {len(v['L'])} points, {len(v['jumps'])} jumps")
        for j in v["jumps"]:
            md.append(f"  {j['L_from']:.0f}->{j['L_to']:.0f} km: max jump {j['max_norm_jump']:.3f}; R {j['R_from']:.4e} -> "
                      f"{j['R_to']:.4e}; branch crossing={j['branch_crossing']} (A persists {j['branchA_persists']}, "
                      f"B persists {j['branchB_persists']}); dlogR deviation={j['dlogR_deviation']}; "
                      f"R discontinuous={j['R_discontinuous']}")
    v = out["legacy_1e9"]
    md += ["", f"legacy n_X=1e9, dense scan 140..160 km: {len(v['jumps'])} jumps"]
    for j in v["jumps"]:
        md.append(f"  {j['L_from']:.0f}->{j['L_to']:.0f} km: max jump {j['max_norm_jump']:.3f}; branch crossing="
                  f"{j['branch_crossing']}; R discontinuous={j['R_discontinuous']}")
    for k, js in v["stored_runs"].items():
        md.append(f"  stored run {k} (0.5 km grid), jumps in 140..160 km: " +
                  (", ".join(f"{j['L_from']:.2f}->{j['L_to']:.2f} ({j['max_norm_jump']:.3f})" for j in js) or "none"))
    open(os.path.join(d, "DENSE_SCAN.txt"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
