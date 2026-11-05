"""Fair "worse" check after an evaluation-numerics change: re-evaluate the OLD reference's parameters with the
CURRENT objective and compare with the NEW reference R at every shared point.
  gap = (R_new_ref - R_current(old params)) / |R_current(old params)|;  worse if gap < -tol_ref(new build).
Also reports how much the old stored R differed from R_current(old params) (pure evaluation change).
Usage: python Optimization/analysis/compare_same_objective.py --old-dir pre_expm1 lim lim_literal
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from src.qkd_opt import get_preset  # noqa: E402
from src.qkd_opt.objective import evaluate  # noqa: E402

args = sys.argv[1:]
old_dir = "pre_expm1"
if args[0] == "--old-dir":
    old_dir, args = args[1], args[2:]
REF = os.path.join(ROOT, "Optimization", "reference")
lines = [f"same-objective comparison: new references vs OLD params ({old_dir}) re-evaluated with the current objective"]
summary = {}
for preset in args:
    pre = get_preset(preset)
    f = jax.jit(jax.vmap(lambda p, L, n: evaluate(p, L, n, pre.consts, pre.flags)["R"]))
    for kind in ("", "_tail"):
        new = json.load(open(os.path.join(REF, f"reference_{preset}{kind}.json")))
        old = json.load(open(os.path.join(REF, old_dir, f"reference_{preset}{kind}.json")))
        tol = max(1e-9, 2.0 * float(new["summary"].get("seed_disagreement_rel_max") or 0.0))
        om = {(p["grid_index"], p["n_X"]): p for p in old["points"]}
        sh = [(p, om[(p["grid_index"], p["n_X"])]) for p in new["points"] if (p["grid_index"], p["n_X"]) in om]
        sh = [(p, q) for p, q in sh if p["R"] > 0 or q["R"] > 0]
        Rre = np.asarray(f(jnp.asarray([q["params"] for _, q in sh]), jnp.asarray([q["L"] for _, q in sh]),
                           jnp.asarray([q["n_X"] for _, q in sh])))
        Rnew = np.array([p["R"] for p, _ in sh])
        Rold = np.array([q["R"] for _, q in sh])
        gap = (Rnew - Rre) / np.abs(Rre)
        evalchg = (Rre - Rold) / np.abs(Rold)
        worse = [(sh[i][0]["L"], sh[i][0]["n_X"], gap[i]) for i in np.where(gap < -tol)[0]]
        name = f"{preset}{kind or '_main'}"
        summary[name] = dict(n=len(sh), tol_ref=tol, min_gap=float(gap.min()), max_gap=float(gap.max()),
                             n_worse=len(worse), worse=worse, eval_change_min=float(evalchg.min()),
                             eval_change_max=float(evalchg.max()))
        lines.append(f"{name}: {len(sh)} shared points, tol_ref {tol:.2e}; gap (new ref vs old params under current "
                     f"objective) min {gap.min():+.2e} max {gap.max():+.2e}; WORSE {len(worse)}; evaluation change of the "
                     f"old params (current vs stored R) min {evalchg.min():+.2e} max {evalchg.max():+.2e}")
        for L, n, g in worse:
            lines.append(f"    WORSE L={L:.3f} nX={n:.0e} gap={g:+.3e}")
d = os.path.join(ROOT, "results", f"compare_same_objective_vs_{old_dir}")
os.makedirs(d, exist_ok=True)
json.dump(summary, open(os.path.join(d, "compare_same_objective.json"), "w"), indent=1, default=float)
open(os.path.join(d, "COMPARE_SAME_OBJECTIVE.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
