"""Build the trusted reference ("ground truth") for one physics preset.

Main reference (default): the legacy 400-point grid linspace(0, 200, 400) subsampled every 20th
point plus the last point (21 fiber lengths, ~10 km apart, identical L values to the legacy runs)
x all six n_X.

Tail reference (--tail): every grid point of linspace(0, 200, 400) with
    horizon_ref(n_X) - 15 km <= L <= horizon_ref(n_X) + 1 km
for each n_X, using the horizons of the main reference (which must exist). This is where the
basins are narrowest and the legacy methods failed.

For each point the reference is the best of
  * two independent dense cold solves: 2^17 scrambled-Sobol points (seeds 0 and 1), the 64 best
    diverse candidates refined by the exact-Hessian Newton solver;
  * (legacy preset only) Newton refinement started from each legacy run's stored parameters
    (old DA, old Hessian, old Hessian_FoR) -- so the reference is never worse than a basin those
    methods found.
The winning source and the disagreement between the two seeds are recorded per point.

Gradient tiers (certificate gradient norm g = inf-norm of the gradient of the scaled objective in theta),
for points with R > 0:
  g > CLIFF_GTOL = 1e-3          -> status "...|cliff_edge": excluded from the "within 1e-6" regret score,
                                    still listed (eps_sec = kappa*l discontinuity / R -> 0 scaling near the horizon)
  NOISE_GTOL = 1e-6 < g <= 1e-3  -> status "...|noise_floor": scored normally, counted and listed

Neighbour continuation: after the Sobol (and legacy-start) solves, the points of each n_X are swept in
increasing L ("cont_up") and then decreasing L ("cont_down"); each point is warm-started from the best
parameters known at the previous grid point of that sweep (best of all candidates so far, including the
sweep's own previous result). Both sweeps are candidate sources; the reference is the best candidate.
The summary reports per-source win counts and every point where a continuation candidate beats the best
Sobol candidate (relative gain > 1e-12).

Seeds: main reference 0 1; tail reference 2 3 by default (independent of the main reference), and
the tail build reports the max relative R difference at grid points shared with the main reference.

Main reference horizon: for every n_X, bisection (25 steps) on the sign of max_p l between the
last reference L with l > 0 and the next one, each step a fresh dense cold solve (2^15 Sobol,
top-32) plus the previous bracket solution as an extra start.

Usage: python Optimization/reference/build_reference.py --preset lim|lim_literal|legacy [--tail] [--quick]
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
from src.qkd_opt.optimize import optimize_batch, warm_solve, _finalise, eval_starts  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from src.qkd_opt import reparam  # noqa: E402
from src.qkd_opt.provenance import stamp, to_jsonable  # noqa: E402
import legacy_runs  # noqa: E402

NX = legacy_runs.NX
GRID = legacy_runs.GRID
REF_IDX = list(range(0, 400, 20)) + [399]
TAIL_BELOW_KM, TAIL_ABOVE_KM = 15.0, 1.0
CLIFF_GTOL = 1e-3
NOISE_GTOL = 1e-6


def ref_path(preset_name, tail=False, quick=False):
    return os.path.join(ROOT, "Optimization", "reference",
                        f"reference_{preset_name}{'_tail' if tail else ''}{'_QUICK' if quick else ''}.json")


def _score(R, l, n):
    return np.where(np.isfinite(R), R if R > 0 else l / n, -np.inf)


def continuation(L, n, preset, cands):
    """cont_up / cont_down candidates (see module doc). Returns dict name -> finalised arrays [P]."""
    P = len(L)
    best_p = np.zeros((P, 5))
    best_v = np.full(P, -np.inf)
    for c in cands.values():
        for i in range(P):
            v = _score(c["R"][i], c["l"][i], n[i])
            if v > best_v[i]:
                best_v[i], best_p[i] = v, c["params"][i]
    out = {}
    for name, rev in (("cont_up", False), ("cont_down", True)):
        th = np.zeros((P, 5))
        st = np.zeros(P, int)
        it = np.zeros(P, int)
        mode = np.zeros(P, int)
        scale = np.ones(P)
        for nx in np.unique(n):
            ids = np.where(n == nx)[0]
            ids = ids[np.argsort(L[ids])]
            if rev:
                ids = ids[::-1]
            prev = None
            for k, i in enumerate(ids):
                start = best_p[i] if prev is None else prev
                w = warm_solve(reparam.theta_of_params(start[None], preset.consts.mu3), L[i:i + 1], n[i:i + 1], preset)
                th[i], st[i], it[i], mode[i], scale[i] = w["theta"][0], w["status"][0], w["iters"][0], w["mode"][0], w["scale"][0]
                pi = np.asarray(reparam.params_of_theta(jnp.asarray(th[i]), preset.consts.mu3))
                R_i, l_i = eval_starts(jnp.asarray(th[i])[None, None, :], jnp.asarray(L[i:i + 1]),
                                       jnp.asarray(n[i:i + 1]), preset.consts, preset.flags)
                v = _score(float(np.asarray(R_i)[0, 0]), float(np.asarray(l_i)[0, 0]), n[i])
                if v > best_v[i]:
                    best_v[i], best_p[i] = v, pi
                prev = best_p[i]
        out[name] = _finalise(dict(theta=th, status=st, iters=it, mode=mode, scale=scale), L, n, preset)
        print(f"[ref] continuation {name} done", flush=True)
    return out


def solve_points(L, n, idx, preset, n_sobol, top_k, seeds):
    """Dense multi-source solve of points (L[P], n[P]); idx[P] = grid index (for legacy starts)."""
    cands = {}
    for s in seeds:
        t = time.perf_counter()
        cands[f"sobol_seed{s}"] = optimize_batch(L, n, preset, n_sobol=n_sobol, top_k=top_k, seed=s)
        print(f"[ref] seed {s}: {time.perf_counter() - t:.1f}s", flush=True)
    if preset.name == "legacy":
        rows = np.array([NX.index(x) for x in n])
        for name in ("legacy_DA", "legacy_Hessian", "legacy_Hessian_FoR"):
            run = legacy_runs.load(name)
            p0 = run["params"][rows, idx, :]
            w = warm_solve(reparam.theta_of_params(p0, preset.consts.mu3), L, n, preset)
            cands[f"refined_from_{name}"] = _finalise(dict(theta=w["theta"], status=w["status"], iters=w["iters"],
                                                           mode=w["mode"], scale=w["scale"]), L, n, preset)
            print(f"[ref] refined from {name}", flush=True)
    cands.update(continuation(L, n, preset, cands))
    names = list(cands)
    R_all = np.stack([cands[k]["R"] for k in names])
    l_all = np.stack([cands[k]["l"] / n for k in names])
    pos = np.max(R_all, axis=0) > 0
    score = np.where(pos[None, :], R_all, l_all)
    score = np.where(np.isfinite(score), score, -np.inf)
    win = np.argmax(score, axis=0)
    s0, s1 = cands[f"sobol_seed{seeds[0]}"]["R"], cands[f"sobol_seed{seeds[-1]}"]["R"]
    best_R = np.array([cands[names[w]]["R"][i] for i, w in enumerate(win)])
    seed_rel = np.where(pos, np.abs(s0 - s1) / np.maximum(np.abs(best_R), 1e-300), np.nan)
    pts = []
    for i, w in enumerate(win):
        c = cands[names[w]]
        cliff = bool(c["R"][i] > 0 and c["grad_norm_theta"][i] > CLIFF_GTOL)
        noise = bool(c["R"][i] > 0 and NOISE_GTOL < c["grad_norm_theta"][i] <= CLIFF_GTOL)
        pts.append(dict(L=float(L[i]), n_X=float(n[i]), grid_index=int(idx[i]), params=c["params"][i], R=c["R"][i],
                        l=c["l"][i], N=c["N"][i], eps_sec=c["eps_sec"][i],
                        status=c["status"][i] + ("|cliff_edge" if cliff else "") + ("|noise_floor" if noise else ""),
                        cliff_edge=cliff, noise_floor=noise,
                        active_bounds=list(c["active_bounds"][i]), grad_norm_theta=c["grad_norm_theta"][i],
                        hess_eig=c["hess_eig"][i], source=names[w],
                        candidates={k: dict(R=cands[k]["R"][i], l=cands[k]["l"][i]) for k in names},
                        seed_disagreement_rel=seed_rel[i]))
    summary = dict(n_points=len(L), n_positive=int(pos.sum()),
                   source_counts={k: int(np.sum(win == i)) for i, k in enumerate(names)},
                   seed_disagreement_rel_max=float(np.nanmax(seed_rel)) if pos.any() else None,
                   seed_disagreement_rel_median=float(np.nanmedian(seed_rel)) if pos.any() else None,
                   n_points_seed_disagree_gt_1e9=int(np.nansum(seed_rel > 1e-9)),
                   n_cliff_edge=int(sum(p["cliff_edge"] for p in pts)),
                   cliff_edge_points=[dict(L=p["L"], n_X=p["n_X"], R=p["R"], l=p["l"],
                                           grad_norm_theta=p["grad_norm_theta"]) for p in pts if p["cliff_edge"]],
                   n_noise_floor=int(sum(p["noise_floor"] for p in pts)),
                   noise_floor_points=[dict(L=p["L"], n_X=p["n_X"], R=p["R"], l=p["l"],
                                            grad_norm_theta=p["grad_norm_theta"]) for p in pts if p["noise_floor"]],
                   continuation_improves_over_sobol=_cont_gain(cands, names, L, n))
    return pts, summary


def _cont_gain(cands, names, L, n):
    sob = [k for k in names if k.startswith("sobol_seed")]
    out = []
    for i in range(len(L)):
        vs = max(_score(cands[k]["R"][i], cands[k]["l"][i], n[i]) for k in sob)
        for k in ("cont_up", "cont_down"):
            vc = _score(cands[k]["R"][i], cands[k]["l"][i], n[i])
            if np.isfinite(vs) and vc > vs + 1e-12 * abs(vs):
                out.append(dict(L=float(L[i]), n_X=float(n[i]), source=k, R_best_sobol=float(max(cands[q]["R"][i] for q in sob)),
                                R_cont=float(cands[k]["R"][i]), rel_gain=float((vc - vs) / abs(vs))))
    return dict(n_points=len({(o["L"], o["n_X"]) for o in out}), entries=out)


def horizons(pts, preset, n_sobol, top_k, steps):
    out = []
    for nx in NX:
        P = sorted([p for p in pts if p["n_X"] == nx], key=lambda p: p["L"])
        R = np.array([p["R"] for p in P])
        posi = np.where(R > 0)[0]
        if len(posi) == 0 or posi[-1] == len(P) - 1:
            out.append(dict(n_X=nx, horizon_km=None, note="no sign change on reference grid"))
            continue
        k = posi[-1]
        lo, hi, p_lo = P[k]["L"], P[k + 1]["L"], np.asarray(P[k]["params"])
        for step in range(steps):
            mid = 0.5 * (lo + hi)
            r = optimize_batch([mid], [nx], preset, n_sobol=max(n_sobol // 4, 2048), top_k=max(top_k // 2, 4),
                               seed=1000 + step, extra_starts=p_lo[None, None, :])
            if r["l"][0] > 0:
                lo, p_lo = mid, r["params"][0]
            else:
                hi = mid
        out.append(dict(n_X=nx, horizon_km=lo, bracket_km=[lo, hi]))
        print(f"[ref] horizon n_X={nx:.0e}: {lo:.6f} km", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True)
    ap.add_argument("--tail", action="store_true")
    ap.add_argument("--n-sobol", type=int, default=2**17)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="default: 0 1 (main), 2 3 (tail)")
    ap.add_argument("--horizon-steps", type=int, default=25)
    ap.add_argument("--quick", action="store_true", help="tiny budget smoke run (not a reference)")
    a = ap.parse_args()
    if a.seeds is None:
        a.seeds = [2, 3] if a.tail else [0, 1]
    if a.quick:
        a.n_sobol, a.top_k, a.horizon_steps = 2**11, 8, 6
    preset = get_preset(a.preset)
    backend = check_x64()
    t0 = time.perf_counter()

    if a.tail:
        main_ref = json.load(open(ref_path(preset.name, quick=a.quick)))
        hz = {h["n_X"]: h.get("horizon_km") for h in main_ref["horizons"]}
        idx, n = [], []
        for nx in NX:
            h = hz.get(nx)
            if h is None:
                continue
            sel = np.where((GRID >= h - TAIL_BELOW_KM) & (GRID <= h + TAIL_ABOVE_KM))[0]
            idx += sel.tolist()
            n += [nx] * len(sel)
        idx, n = np.array(idx), np.array(n, float)
        L = GRID[idx]
        window = dict(below_horizon_km=TAIL_BELOW_KM, above_horizon_km=TAIL_ABOVE_KM, main_reference_horizons=hz,
                      main_reference_config_hash=main_ref["stamp"]["config_hash"])
    else:
        idx = np.tile(REF_IDX, len(NX))
        L = GRID[idx]
        n = np.repeat(NX, len(REF_IDX)).astype(float)
        window = None

    cfg = dict(script="build_reference.py", preset=preset.name, tail=a.tail, n_sobol=a.n_sobol, top_k=a.top_k,
               seeds=a.seeds, nX=NX, horizon_steps=a.horizon_steps, window=window,
               grid_indices=idx.tolist(), flags=preset.flags._asdict(), consts=preset.consts._asdict())
    st = stamp(cfg)
    pts, summary = solve_points(L, n, idx, preset, a.n_sobol, a.top_k, a.seeds)
    hzs = None if a.tail else horizons(pts, preset, a.n_sobol, a.top_k, a.horizon_steps)
    if a.tail:   # independent-seed overlap check against the main reference
        mref = {(p["grid_index"] if "grid_index" in p else int(np.argmin(np.abs(GRID - p["L"]))), p["n_X"]): p
                for p in main_ref["points"]}
        ov = []
        for p in pts:
            q = mref.get((p["grid_index"], p["n_X"]))
            if q is not None and q["R"] > 0:
                ov.append(dict(L=p["L"], n_X=p["n_X"], R_main=q["R"], R_tail=p["R"],
                               rel_diff=abs(p["R"] - q["R"]) / abs(q["R"]), main_seeds=main_ref["config"]["seeds"],
                               tail_seeds=a.seeds))
        summary["overlap_with_main"] = dict(n=len(ov), max_rel_diff=max((o["rel_diff"] for o in ov), default=None),
                                            points=ov)
    wall = time.perf_counter() - t0
    out = dict(description=__doc__, stamp=st, config=cfg, backend=backend, wall_time_s=wall, points=pts,
               horizons=hzs, summary=summary)
    path = ref_path(preset.name, tail=a.tail, quick=a.quick)
    with open(path, "w") as fh:
        json.dump(to_jsonable(out), fh, indent=1)
    print(json.dumps(to_jsonable(summary), indent=1))
    print(f"[ref] wrote {os.path.relpath(path, ROOT)} in {wall:.1f}s")


if __name__ == "__main__":
    main()
