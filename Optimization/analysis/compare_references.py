"""Step 4: compare rebuilt references with the archived pre-fix ones (Optimization/reference/prefix_eps_fix/).

For each preset and kind (main, tail): relative R difference at every shared point (same grid index and n_X),
lists of points where the new reference is BETTER by > 1e-9 / WORSE by > tol_ref relative (tol_ref = max(1e-9,
2 x seed_disagreement_rel_max of the NEW build); worse by 1e-9 .. tol_ref is listed as INFO, horizon change per n_X (main),
and gradient-tier counts before vs after. "Before" counts apply the current thresholds (cliff_edge g > 1e-3,
noise_floor 1e-6 < g <= 1e-3, R > 0) to the archived certificate gradients, so the counts are comparable.
Flags: any WORSE point, any horizon change > 0.5 km, any shared-point R change > 1e-6 relative.

Usage: python Optimization/analysis/compare_references.py lim [lim_literal ...]
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REF = os.path.join(ROOT, "Optimization", "reference")


def tiers(points):
    c = n = 0
    for p in points:
        if p["R"] > 0:
            g = p["grad_norm_theta"]
            c += g > 1e-3
            n += 1e-6 < g <= 1e-3
    return int(c), int(n)


def key(p):
    return (p.get("grid_index"), p["n_X"])


def main():
    out, flags = {}, []
    lines = []
    for preset in sys.argv[1:]:
        for kind in ("", "_tail"):
            new = json.load(open(os.path.join(REF, f"reference_{preset}{kind}.json")))
            old = json.load(open(os.path.join(REF, "prefix_eps_fix", f"reference_{preset}{kind}.json")))
            om = {key(p): p for p in old["points"]}
            rows, better, worse, info = [], [], [], []
            tol_ref = max(1e-9, 2.0 * float(new["summary"].get("seed_disagreement_rel_max") or 0.0))
            for p in new["points"]:
                q = om.get(key(p))
                if q is None:
                    continue
                if q["R"] > 0 or p["R"] > 0:
                    rel = (p["R"] - q["R"]) / max(abs(q["R"]), 1e-300)
                    rows.append(rel)
                    rec = dict(L=p["L"], n_X=p["n_X"], R_old=q["R"], R_new=p["R"], rel=rel, source_new=p["source"])
                    if rel > 1e-9:
                        better.append(rec)
                    elif rel < -tol_ref:
                        worse.append(rec)
                    elif rel < -1e-9:
                        info.append(rec)
            rows = np.array(rows)
            name = f"{preset}{kind or '_main'}"
            hz = None
            if new.get("horizons") and old.get("horizons"):
                ho = {h["n_X"]: h.get("horizon_km") for h in old["horizons"]}
                hz = {f"{h['n_X']:.0e}": dict(old=ho.get(h["n_X"]), new=h.get("horizon_km"),
                                              diff=None if (h.get("horizon_km") is None or ho.get(h["n_X"]) is None)
                                              else h["horizon_km"] - ho[h["n_X"]]) for h in new["horizons"]}
            cb, nb = tiers(old["points"])
            ca, na = tiers(new["points"])
            out[name] = dict(n_shared=int(rows.size), max_abs_rel=float(np.max(np.abs(rows))) if rows.size else None,
                             n_better=len(better), n_worse=len(worse), better=better, worse=worse, horizons=hz,
                             tol_ref=tol_ref, n_info=len(info), info=info,
                             cliff_before=cb, noise_before=nb, cliff_after=ca, noise_after=na,
                             new_stamp=new["stamp"]["git_commit"], old_stamp=old["stamp"]["git_commit"],
                             new_physics=new["stamp"]["physics_source_hash"], old_physics=old["stamp"]["physics_source_hash"],
                             continuation=new["summary"].get("continuation_improves_over_sobol"),
                             source_counts=new["summary"]["source_counts"])
            if worse:
                flags.append(f"{name}: {len(worse)} points WORSE than pre-fix reference")
            if rows.size and np.max(np.abs(rows)) > 1e-6:
                flags.append(f"{name}: shared-point R change up to {np.max(np.abs(rows)):.2e} relative (> 1e-6)")
            for k, v in (hz or {}).items():
                if v["diff"] is not None and abs(v["diff"]) > 0.5:
                    flags.append(f"{name}: horizon n_X={k} moved by {v['diff']:+.3f} km")
            o = out[name]
            lines.append(f"{name}: new git {o['new_stamp']} phys {o['new_physics']} vs old git {o['old_stamp']} phys {o['old_physics']}")
            lines.append(f"  shared points {o['n_shared']}, max |rel R diff| {o['max_abs_rel']}, better {o['n_better']}, "
                         f"worse (> tol_ref {tol_ref:.3e}) {o['n_worse']}, INFO (worse by 1e-9..tol_ref) {o['n_info']}")
            for b in better:
                lines.append(f"    BETTER L={b['L']:.3f} nX={b['n_X']:.0e} R_old={b['R_old']:.10e} R_new={b['R_new']:.10e} rel={b['rel']:+.3e} source={b['source_new']}")
            for w in worse:
                lines.append(f"    WORSE  L={w['L']:.3f} nX={w['n_X']:.0e} R_old={w['R_old']:.10e} R_new={w['R_new']:.10e} rel={w['rel']:+.3e}")
            for w in info:
                lines.append(f"    INFO   L={w['L']:.3f} nX={w['n_X']:.0e} R_old={w['R_old']:.10e} R_new={w['R_new']:.10e} rel={w['rel']:+.3e} (within tol_ref)")
            lines.append(f"  tiers (current thresholds): cliff_edge {cb} -> {ca}; noise_floor {nb} -> {na}")
            lines.append(f"  source wins: {o['source_counts']}")
            c = o["continuation"] or {}
            lines.append(f"  continuation improves over best Sobol at {c.get('n_points')} points")
            for e in (c.get("entries") or []):
                lines.append(f"    {e['source']} L={e['L']:.3f} nX={e['n_X']:.0e} R_sobol={e['R_best_sobol']:.10e} R_cont={e['R_cont']:.10e} gain={e['rel_gain']:.3e}")
            if hz:
                lines.append("  horizons km (old -> new): " + "; ".join(
                    f"{k}: {v['old']:.4f} -> {v['new']:.4f} ({v['diff']:+.4f})" for k, v in hz.items() if v["diff"] is not None))
    lines.append("FLAGS: " + ("; ".join(flags) if flags else "none"))
    d = os.path.join(ROOT, "results", "compare_references")
    os.makedirs(d, exist_ok=True)
    json.dump(dict(results=out, flags=flags), open(os.path.join(d, "compare_references.json"), "w"), indent=1, default=float)
    open(os.path.join(d, "COMPARE_REFERENCES.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
