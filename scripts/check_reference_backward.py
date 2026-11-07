"""Read-only check: does an independent backward warm sweep beat the stored main reference anywhere?

Motivation: STATUS.md section 10d (branch qml-surrogate) found that a single forward warm sweep can stay on a worse
branch past a crossover (12 silent switches, 58 sub-optimal labels, up to 6e-4). The reference
(Optimization/reference/reference_lim{,_tail}.json) already takes the best of two cold Sobol solves and forward /
backward neighbour continuation per point; this script stress-tests that with a dense backward sweep.

Method: optimize_sweep (preset lim, default sweep settings, global check every 10 grid points) on a DESCENDING grid
= every reference L value (the legacy 400-point grid) united with a 1/8-km lattice on [0, 200] km, for the six
reference n_X, without horizon refinement. At every reference point with R_ref > 0, compare R_backward with R_ref;
flag rel = (R_bw - R_ref)/R_ref > tol_ref (1e-9, from the reference summaries).
Nothing in Optimization/ or the reference JSONs is modified. Output: results/check_reference_backward_<git>_<cfg>/.

Usage: JAX_PLATFORMS=cpu .venv/bin/python scripts/check_reference_backward.py
"""
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
from src.qkd_opt import get_preset, check_x64  # noqa: E402
from src.qkd_opt.optimize import optimize_sweep  # noqa: E402
from src.qkd_opt.provenance import stamp, to_jsonable, config_hash, power_state  # noqa: E402


def main():
    pw = power_state()
    if pw["source"] != "AC":
        sys.exit(f"refusing to run: power source {pw['source']!r}")
    check_x64()
    pre = get_preset("lim")
    refs = {t: json.load(open(os.path.join(ROOT, "Optimization", "reference", f"reference_lim{s}.json")))
            for t, s in (("main", ""), ("tail", "_tail"))}
    nx_list = sorted({p["n_X"] for r in refs.values() for p in r["points"]})
    ref_L = np.array(sorted({p["L"] for r in refs.values() for p in r["points"]}))
    G = np.unique(np.concatenate([np.linspace(0.0, 200.0, 400), np.arange(0, 1601) / 8.0, ref_L]))
    cfg = dict(script="scripts/check_reference_backward.py", preset=pre.name, nX=nx_list, n_grid=int(G.size),
               grid="linspace(0,200,400) U 1/8-km lattice U reference L, descending", sweep="optimize_sweep defaults, refine_horizon=False",
               tol_ref={t: r["summary"]["tol_ref"] for t, r in refs.items()})
    st = stamp(cfg)
    t0 = time.perf_counter()
    bw = optimize_sweep(G[::-1], nx_list, pre, refine_horizon=False)
    wall = time.perf_counter() - t0
    Gd = G[::-1]
    pos = {(float(nx), float(l_)): (i, j) for i, nx in enumerate(nx_list) for j, l_ in enumerate(Gd)}
    R_bw, l_bw = np.asarray(bw["R"]), np.asarray(bw["l"])

    rows, flagged = [], []
    for tag, r in refs.items():
        tol = r["summary"]["tol_ref"]
        for p in r["points"]:
            i, j = pos[(float(p["n_X"]), float(p["L"]))]
            rb = float(R_bw[i, j])
            row = dict(set=tag, L=p["L"], n_X=p["n_X"], R_ref=p["R"], R_backward=rb, cliff_edge=bool(p["cliff_edge"]))
            if p["R"] > 0:
                row["rel"] = (rb - p["R"]) / p["R"]
                if row["rel"] > tol:
                    flagged.append(row)
            rows.append(row)
    scored = [x for x in rows if "rel" in x]
    rel = np.array([x["rel"] for x in scored])
    summ = dict(n_points=len(rows), n_R_positive=len(scored),
                n_R_positive_non_cliff=sum(not x["cliff_edge"] for x in scored),
                n_backward_better_than_tol_ref=len(flagged),
                max_rel_backward_minus_ref=float(rel.max()), median_rel=float(np.median(rel)),
                n_backward_worse_than_1e9=int((rel < -1e-9).sum()),
                worst_backward_deficit=float(rel.min()),
                n_backward_events=len(bw["events"]),
                n_backward_switch_events_objR=sum(1 for e in bw["events"] if e["type"] == "branch_switch" and e.get("objective") == "R"))
    out = dict(stamp=st, config=cfg, power_at_start=pw, sweep_wall_s=wall, summary=summ, flagged=flagged, rows=rows,
               backward_events=bw["events"])
    od = os.path.join(ROOT, "results", f"check_reference_backward_{st['git_commit']}_{config_hash(cfg)}")
    os.makedirs(od, exist_ok=True)
    with open(os.path.join(od, "check.json"), "w") as fh:
        json.dump(to_jsonable(out), fh, indent=1)
    md = ["# Backward-sweep check of the main reference (preset lim)", "",
          f"git {st['git_commit']} | cfg {st['config_hash']} | phys {st['physics_source_hash']} | power {pw['source']} | sweep {wall:.0f} s", "",
          f"Grid: {G.size} L values (descending) × {len(nx_list)} n_X. Reference points: {summ['n_points']} "
          f"({summ['n_R_positive']} with R_ref > 0, {summ['n_R_positive_non_cliff']} of them not cliff_edge).", "",
          f"**Backward solve better than the reference by > tol_ref (1e-9): {len(flagged)} points.**", "",
          f"Max (R_bw - R_ref)/R_ref = {summ['max_rel_backward_minus_ref']:.3e}; median {summ['median_rel']:.3e}; "
          f"backward worse than the reference by > 1e-9 at {summ['n_backward_worse_than_1e9']} points "
          f"(worst {summ['worst_backward_deficit']:.3e}). Backward sweep logged {summ['n_backward_switch_events_objR']} "
          f"branch switches with objective R.", ""]
    if flagged:
        md += ["| set | L km | n_X | R_ref | R_backward | rel. improvement | cliff_edge |", "|---|---|---|---|---|---|---|"]
        md += [f"| {x['set']} | {x['L']:.3f} | {x['n_X']:.0e} | {x['R_ref']:.6e} | {x['R_backward']:.6e} | {x['rel']:.3e} | {x['cliff_edge']} |"
               for x in sorted(flagged, key=lambda x: -x["rel"])]
    with open(os.path.join(od, "REPORT.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))
    print("wrote", os.path.relpath(od, ROOT))


if __name__ == "__main__":
    main()
