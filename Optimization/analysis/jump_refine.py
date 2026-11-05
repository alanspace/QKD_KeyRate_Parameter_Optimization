"""Refine every jump of the newest dense_scan at 0.1 km: warm continuation from the optimum at L_from across
[L_from, L_to] in 10 steps (each step: warm solve from the previous step's optimum, plus a cold 2^14 solve; best
kept). Reports per jump the largest single 0.1 km parameter step (normalised by the box) and the largest
relative R step. A genuine branch switch shows one large step at 0.1 km; continuous drift shows ~1/10 of the
1 km jump per step. Usage: python Optimization/analysis/jump_refine.py
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
from src.qkd_opt import LIM, reparam  # noqa: E402
from src.qkd_opt.optimize import optimize_batch, warm_solve  # noqa: E402
from src.qkd_opt.provenance import stamp, to_jsonable  # noqa: E402

BOX = np.array([reparam.MU1_MAX, reparam.MU2_MAX, 1.0, 1.0, 1.0])
src = sorted(glob.glob(os.path.join(ROOT, "results", "dense_scan_*", "dense_scan.json")), key=os.path.getmtime)[-1]
ds = json.load(open(src))
st = stamp(dict(script="jump_refine.py", source=os.path.relpath(src, ROOT)))
rows, lines = [], [f"jump refinement at 0.1 km of {os.path.relpath(src, ROOT)}, git {st['git_commit']}"]
for k, v in ds["lim"].items():
    for j in v.get("jumps", []):
        nx = float(k)
        Ls = np.linspace(j["L_from"], j["L_to"], 11)
        p = np.asarray(j["params_from"])
        Ps, Rs = [p], [j["R_from"]]
        for L in Ls[1:]:
            w = warm_solve(reparam.theta_of_params(p[None], LIM.consts.mu3), [L], [nx], LIM)
            pw = np.asarray(reparam.params_of_theta(w["theta"][0], LIM.consts.mu3))
            c = optimize_batch([L], [nx], LIM, n_sobol=2**14, top_k=16, seed=8, extra_starts=pw[None, None, :])
            p, R = c["params"][0], float(c["R"][0])
            Ps.append(p); Rs.append(R)
        Ps, Rs = np.array(Ps), np.array(Rs)
        step = np.abs(np.diff(Ps / BOX, axis=0)).max(axis=1)
        pos = (Rs[:-1] > 0) & (Rs[1:] > 0)
        rstep = np.where(pos, np.abs(np.diff(Rs)) / np.maximum(Rs[:-1], 1e-300), np.nan)
        rec = dict(n_X=nx, L_from=j["L_from"], L_to=j["L_to"], jump_1km=j["max_norm_jump"], max_step_0p1km=float(step.max()),
                   ratio=float(step.max() / j["max_norm_jump"]), max_rel_R_step=float(np.nanmax(rstep)) if pos.any() else None)
        rows.append(rec)
        lines.append(f"  n_X={k} {j['L_from']:.0f}->{j['L_to']:.0f} km: 1 km jump {j['max_norm_jump']:.3f}; largest 0.1 km step "
                     f"{rec['max_step_0p1km']:.4f} ({100 * rec['ratio']:.0f}% of the 1 km jump); largest relative R step per 0.1 km "
                     f"{rec['max_rel_R_step']}")
d = os.path.join(ROOT, "results", f"jump_refine_{st['git_commit']}_{st['config_hash']}")
os.makedirs(d, exist_ok=True)
json.dump(to_jsonable(dict(stamp=st, rows=rows)), open(os.path.join(d, "jump_refine.json"), "w"), indent=1)
open(os.path.join(d, "JUMP_REFINE.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
