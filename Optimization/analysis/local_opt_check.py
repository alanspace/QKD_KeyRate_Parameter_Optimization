"""Local-optimality perturbation check of a tail reference.

For the last 12 points with R > 0 of every n_X in reference_<preset>_tail.json: 1e5 random perturbations
theta' = theta* + s z (z ~ N(0, I_5), s log-uniform in [1e-6, 1e-1], numpy PCG64 seed 99), evaluated with
jit+vmap. Reports the best relative improvement (R(theta') - R*)/R* per point and every point where it
exceeds tol_ref = max(1e-9, 2 x seed_disagreement_rel_max of that reference build).

Usage: python Optimization/analysis/local_opt_check.py --preset lim
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

from src.qkd_opt import get_preset, check_x64, reparam  # noqa: E402
from src.qkd_opt.objective import evaluate  # noqa: E402
from src.qkd_opt.provenance import stamp, to_jsonable  # noqa: E402

N_PERT, LAST, CHUNK = 100_000, 12, 20_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="lim")
    a = ap.parse_args()
    pre = get_preset(a.preset)
    check_x64()
    rpath = os.path.join(ROOT, "Optimization", "reference", f"reference_{pre.name}_tail.json")
    ref = json.load(open(rpath))
    tol_ref = max(1e-9, 2.0 * float(ref["summary"]["seed_disagreement_rel_max"] or 0.0))
    cfg = dict(script="local_opt_check.py", preset=pre.name, reference=os.path.relpath(rpath, ROOT),
               reference_config_hash=ref["stamp"]["config_hash"], n_perturbations=N_PERT, last_points_per_nX=LAST,
               scales=[1e-6, 1e-1], seed=99, tol_ref=tol_ref)
    st = stamp(cfg)
    f = jax.jit(jax.vmap(lambda th, L, n: evaluate(reparam.params_of_theta(th, pre.consts.mu3), L, n,
                                                   pre.consts, pre.flags)["R"], in_axes=(0, None, None)))
    rng = np.random.default_rng(99)
    rows = []
    t0 = time.perf_counter()
    for nx in sorted({p["n_X"] for p in ref["points"]}):
        pts = sorted([p for p in ref["points"] if p["n_X"] == nx and p["R"] > 0], key=lambda p: p["L"])[-LAST:]
        for p in pts:
            th0 = reparam.theta_of_params(np.asarray(p["params"])[None], pre.consts.mu3)[0]
            R0 = float(f(jnp.asarray(th0[None]), p["L"], p["n_X"])[0])
            best, best_s = -np.inf, None
            for s_ in range(0, N_PERT, CHUNK):
                sc = 10.0 ** rng.uniform(-6, -1, size=(CHUNK, 1))
                th = th0[None] + sc * rng.standard_normal((CHUNK, 5))
                Rp = np.asarray(f(jnp.asarray(th), p["L"], p["n_X"]))
                Rp = np.where(np.isfinite(Rp), Rp, -np.inf)
                k = int(np.argmax(Rp))
                if Rp[k] > best:
                    best, best_s = float(Rp[k]), float(sc[k, 0])
            gain = (best - R0) / abs(R0)
            rows.append(dict(L=p["L"], n_X=nx, R_ref=p["R"], R_ref_reeval=R0, best_perturbed=best, rel_gain=gain,
                             scale_of_best=best_s, status=p["status"], exceeds_tol_ref=bool(gain > tol_ref)))
        print(f"[localopt] n_X={nx:.0e}: max gain {max(r['rel_gain'] for r in rows if r['n_X'] == nx):.3e}", flush=True)
    ex = [r for r in rows if r["exceeds_tol_ref"]]
    out = dict(stamp=st, config=cfg, points=rows, n_points=len(rows), n_exceeding=len(ex),
               max_rel_gain=max(r["rel_gain"] for r in rows), wall_time_s=time.perf_counter() - t0)
    d = os.path.join(ROOT, "results", f"local_opt_check_{pre.name}_{st['git_commit']}_{st['config_hash']}")
    os.makedirs(d, exist_ok=True)
    json.dump(to_jsonable(out), open(os.path.join(d, "local_opt_check.json"), "w"), indent=1)
    lines = [f"local-optimality perturbation check, preset {pre.name}, git {st['git_commit']}, config {st['config_hash']}",
             f"points: {len(rows)} (last {LAST} with R>0 per n_X of the tail reference); {N_PERT} perturbations each, "
             f"scales 1e-6..1e-1 (log-uniform), tol_ref = {tol_ref:.3e}; wall {out['wall_time_s']:.0f} s",
             f"max relative improvement found: {out['max_rel_gain']:.3e}; points with improvement > tol_ref: {len(ex)}"]
    for r in ex:
        lines.append(f"  EXCEEDS L={r['L']:.3f} nX={r['n_X']:.0e} R_ref={r['R_ref']:.10e} best={r['best_perturbed']:.10e} "
                     f"gain={r['rel_gain']:.3e} scale={r['scale_of_best']:.1e} status={r['status']}")
    open(os.path.join(d, "LOCAL_OPT_CHECK.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
