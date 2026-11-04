"""Build the trusted reference ("ground truth") for one physics preset.

Points: the legacy 400-point grid linspace(0, 200, 400) subsampled every 20th point plus the last
point (21 fiber lengths, ~10 km apart, identical L values to the legacy runs) x all six n_X.

For each point the reference is the best of
  * two independent dense cold solves: 2^17 scrambled-Sobol points (seeds 0 and 1), the 64 best
    diverse candidates refined by the exact-Hessian Newton solver;
  * (legacy preset only) Newton refinement started from each legacy run's stored parameters
    (old DA, old Hessian, old Hessian_FoR) -- so the reference is never worse than a basin those
    methods found.
The winning source and the seed-0 / seed-1 disagreement are recorded per point.

Reference horizon: for every n_X, bisection (25 steps) on the sign of max_p l between the last
reference L with l > 0 and the next one, each step a fresh dense cold solve (2^15 Sobol, top-32)
plus the previous bracket solution as an extra start.

Usage: python Optimization/reference/build_reference.py --preset legacy|lim_exact [--quick]
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "Optimization", "benchmark"))

import numpy as np  # noqa: E402

from src.qkd_opt import get_preset, check_x64  # noqa: E402
from src.qkd_opt.optimize import optimize_batch, warm_solve, certify_batch  # noqa: E402
from src.qkd_opt import reparam  # noqa: E402
from src.qkd_opt.provenance import stamp, to_jsonable  # noqa: E402
import legacy_runs  # noqa: E402

NX = legacy_runs.NX
GRID = legacy_runs.GRID
REF_IDX = list(range(0, 400, 20)) + [399]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True)
    ap.add_argument("--n-sobol", type=int, default=2**17)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--horizon-steps", type=int, default=25)
    ap.add_argument("--quick", action="store_true", help="tiny budget smoke run (not a reference)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.quick:
        a.n_sobol, a.top_k, a.horizon_steps = 2**11, 8, 6
    preset = get_preset(a.preset)
    backend = check_x64()
    cfg = dict(script="build_reference.py", preset=preset.name, n_sobol=a.n_sobol, top_k=a.top_k,
               seeds=a.seeds, ref_idx=REF_IDX, nX=NX, horizon_steps=a.horizon_steps,
               flags=preset.flags._asdict(), consts=preset.consts._asdict())
    st = stamp(cfg)
    t0 = time.perf_counter()

    Li = GRID[REF_IDX]
    L = np.tile(Li, len(NX))
    n = np.repeat(NX, len(Li))
    P = len(L)
    cands = {}
    for s in a.seeds:
        t = time.perf_counter()
        r = optimize_batch(L, n, preset, n_sobol=a.n_sobol, top_k=a.top_k, seed=s)
        cands[f"sobol_seed{s}"] = r
        print(f"[ref] seed {s}: {time.perf_counter() - t:.1f}s", flush=True)
    if preset.name == "legacy":
        for name in ("legacy_DA", "legacy_Hessian", "legacy_Hessian_FoR"):
            run = legacy_runs.load(name)
            p0 = run["params"][:, REF_IDX, :].reshape(-1, 5)
            th0 = reparam.theta_of_params(p0, preset.consts.mu3)
            w = warm_solve(th0, L, n, preset)
            # certify at the refined theta
            sol = dict(theta=w["theta"], status=w["status"], iters=w["iters"], mode=w["mode"], scale=w["scale"])
            from src.qkd_opt.optimize import _finalise
            cands[f"refined_from_{name}"] = _finalise(sol, L, n, preset)
            print(f"[ref] refined from {name}", flush=True)

    names = list(cands)

    def value(r):  # objective actually compared: R if any candidate has R>0 else l/n_X
        return r["R"], r["l"] / n
    R_all = np.stack([cands[k]["R"] for k in names])
    l_all = np.stack([cands[k]["l"] / n for k in names])
    pos = np.max(R_all, axis=0) > 0
    score = np.where(pos[None, :], R_all, l_all)
    score = np.where(np.isfinite(score), score, -np.inf)
    win = np.argmax(score, axis=0)
    best = {k: np.stack([cands[names[w]][k][i] for i, w in enumerate(win)])
            for k in ("params", "R", "l", "N", "eps_sec", "grad_norm_theta", "hess_eig", "theta")}
    best["status"] = [cands[names[w]]["status"][i] for i, w in enumerate(win)]
    best["active_bounds"] = [cands[names[w]]["active_bounds"][i] for i, w in enumerate(win)]
    best["source"] = [names[w] for w in win]
    s0, s1 = cands[f"sobol_seed{a.seeds[0]}"]["R"], cands[f"sobol_seed{a.seeds[-1]}"]["R"]
    seed_rel = np.where(pos, np.abs(s0 - s1) / np.maximum(np.abs(best["R"]), 1e-300), np.nan)

    # reference horizon per n_X
    horizons = []
    Rm = best["R"].reshape(len(NX), len(Li))
    Pm = best["params"].reshape(len(NX), len(Li), 5)
    for i, nx in enumerate(NX):
        posi = np.where(Rm[i] > 0)[0]
        if len(posi) == 0 or posi[-1] == len(Li) - 1:
            horizons.append(dict(n_X=nx, horizon_km=None, note="no sign change on reference grid"))
            continue
        k = posi[-1]
        lo, hi, p_lo = Li[k], Li[k + 1], Pm[i, k]
        for step in range(a.horizon_steps):
            mid = 0.5 * (lo + hi)
            r = optimize_batch([mid], [nx], preset, n_sobol=max(a.n_sobol // 4, 2048), top_k=max(a.top_k // 2, 4),
                               seed=1000 + step, extra_starts=p_lo[None, None, :])
            if r["l"][0] > 0:
                lo, p_lo = mid, r["params"][0]
            else:
                hi = mid
        horizons.append(dict(n_X=nx, horizon_km=lo, bracket_km=[lo, hi]))
        print(f"[ref] horizon n_X={nx:.0e}: {lo:.6f} km", flush=True)

    wall = time.perf_counter() - t0
    out = dict(
        description=__doc__, stamp=st, config=cfg, backend=backend, wall_time_s=wall,
        points=[dict(L=float(L[j]), n_X=float(n[j]), params=best["params"][j], R=best["R"][j], l=best["l"][j],
                     N=best["N"][j], eps_sec=best["eps_sec"][j], status=best["status"][j],
                     active_bounds=list(best["active_bounds"][j]), grad_norm_theta=best["grad_norm_theta"][j],
                     hess_eig=best["hess_eig"][j], source=best["source"][j],
                     candidates={k: dict(R=cands[k]["R"][j], l=cands[k]["l"][j]) for k in names},
                     seed_disagreement_rel=seed_rel[j]) for j in range(P)],
        horizons=horizons,
        summary=dict(
            n_points=P, n_positive=int(pos.sum()),
            source_counts={k: int(np.sum(win == i)) for i, k in enumerate(names)},
            seed_disagreement_rel_max=float(np.nanmax(seed_rel)),
            seed_disagreement_rel_median=float(np.nanmedian(seed_rel)),
            n_points_seed_disagree_gt_1e9=int(np.nansum(seed_rel > 1e-9)),
        ),
    )
    path = a.out or os.path.join(ROOT, "Optimization", "reference",
                                 f"reference_{preset.name}{'_QUICK' if a.quick else ''}.json")
    with open(path, "w") as fh:
        json.dump(to_jsonable(out), fh, indent=1)
    print(json.dumps(to_jsonable(out["summary"]), indent=1))
    print(f"[ref] wrote {os.path.relpath(path, ROOT)} in {wall:.1f}s")


if __name__ == "__main__":
    main()
