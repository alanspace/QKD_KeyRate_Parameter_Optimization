"""Amendment 4 acceptance: re-check the known sub-optimal labels (STATUS 10d: 58; STATUS 11(c): 19) against cold solves.

The known points are recorded as stretches, not as a point list. This script therefore re-derives them:
cold solves (optimize_batch, 2^14 Sobol, top-16, seed 11, as in STATUS 10d) at every lattice label with R > 0 inside
each stretch widened by +-MARGIN km. A label is "originally sub-optimal" if the cold solve beats it by > 1e-9 relative.
For every point the repaired label is then compared with the same cold solve. PASS per point: the repaired label is not
worse than the cold solve by > 1e-9 relative.

Usage: .venv/bin/python qml/check_known_suboptimal.py --repaired qml/data/repaired/labels_repaired_<...>.npz
Output: results/qml_suboptcheck_<git>_<cfg>/{check.json,CHECK.md}. Exit code 1 if any point fails.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from qml import common as C  # noqa: E402

# (n_X, km_lo, km_hi, expected count of sub-optimal labels, source)
KNOWN = [
    (1e6, 138.53, 138.75, 8, "STATUS 10d"),
    (3e6, 138.94, 139.16, 8, "STATUS 10d"),
    (10 ** 6.5, 138.91, 139.22, 11, "STATUS 10d"),
    (10 ** 7.5, 98.75, 99.38, 6, "STATUS 10d"),
    (10 ** 7.5, 136.84, 137.19, 12, "STATUS 10d"),
    (1e8, 92.25, 92.38, 2, "STATUS 10d"),
    (10 ** 8.5, 84.88, 85.25, 4, "STATUS 10d"),
    (5e8, 82.00, 82.25, 3, "STATUS 10d"),
    (1e9, 77.63, 77.75, 2, "STATUS 10d"),
    (2e9, 73.13, 73.25, 2, "STATUS 10d"),
    (1e7, 105.375, 105.938, 19, "STATUS 11(c)"),
]
MARGIN = 1.0
TOL = 1e-9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repaired", required=True)
    ap.add_argument("--original", default=None)
    a = ap.parse_args()
    pw = C.require_ac("check_known_suboptimal")
    import jax
    jax.config.update("jax_enable_x64", True)
    from src.qkd_opt import get_preset, check_x64
    from qml.cold import optimize_sliced
    from src.qkd_opt.provenance import to_jsonable
    check_x64()
    pre = get_preset("lim")
    orig = a.original or sorted(glob.glob(os.path.join(C.DATA, "labels_*.npz")))[0]
    d0 = dict(np.load(orig, allow_pickle=False))
    d1 = dict(np.load(a.repaired, allow_pickle=False))
    key = lambda d: {(round(float(x), 6), round(float(y), 9)): i for i, (x, y) in enumerate(zip(np.log10(d["n_X"]), d["L"]))}
    k1 = key(d1)

    rows, idx0, win_of = [], [], []
    for w, (nx, lo, hi, n_exp, srcname) in enumerate(KNOWN):
        m = np.where((np.abs(np.log10(d0["n_X"]) - np.log10(nx)) < 1e-6) & d0["on_lattice"] & (d0["R"] > 0)
                     & (d0["L"] >= lo - MARGIN) & (d0["L"] <= hi + MARGIN))[0]
        idx0 += list(m); win_of += [w] * len(m)
    idx0 = np.array(idx0); win_of = np.array(win_of)
    idx1 = np.array([k1[(round(float(np.log10(d0["n_X"][i])), 6), round(float(d0["L"][i]), 9))] for i in idx0])
    t0 = time.perf_counter()
    cold = optimize_sliced(d0["L"][idx0], d0["n_X"][idx0], pre, n_sobol=2 ** 14, top_k=16, seed=11, tag="check")
    cold_s = time.perf_counter() - t0
    Rc = np.asarray(cold["R"]); R0 = d0["R"][idx0]; R1 = d1["R"][idx1]
    rel0 = (Rc - R0) / np.abs(R0)
    rel1 = (Rc - R1) / np.where(R1 != 0, np.abs(R1), np.nan)
    for j, (i0, i1) in enumerate(zip(idx0, idx1)):
        w = KNOWN[win_of[j]]
        rows.append(dict(window=int(win_of[j]), n_X=float(d0["n_X"][i0]), L=float(d0["L"][i0]), split=str(d0["split"][i0]),
                         in_stretch=bool(w[1] - 1e-9 <= d0["L"][i0] <= w[2] + 1e-9),
                         R_orig=float(R0[j]), R_repaired=float(R1[j]), R_cold=float(Rc[j]),
                         cold_gain_vs_orig=float(rel0[j]), cold_gain_vs_repaired=float(rel1[j]),
                         orig_suboptimal=bool(rel0[j] > TOL), repaired_flag=bool(d1["repaired"][i1]),
                         passed=bool(np.isfinite(rel1[j]) and rel1[j] <= TOL)))
    sub = [r for r in rows if r["orig_suboptimal"]]
    per_win = []
    for w, (nx, lo, hi, n_exp, srcname) in enumerate(KNOWN):
        s = [r for r in sub if r["window"] == w]
        per_win.append(dict(n_X=nx, stretch=[lo, hi], source=srcname, expected=n_exp, found=len(s),
                            found_in_stretch=sum(r["in_stretch"] for r in s),
                            max_orig_deficit=max([r["cold_gain_vs_orig"] for r in s], default=0.0),
                            n_fail_after_repair=sum(not r["passed"] for r in s),
                            n_checked_in_window=sum(r["window"] == w for r in rows),
                            n_fail_any_point_in_window=sum((not r["passed"]) for r in rows if r["window"] == w)))
    res = dict(config=dict(script="qml/check_known_suboptimal.py", original=os.path.relpath(orig, ROOT),
                           repaired=os.path.relpath(a.repaired, ROOT), repaired_sha=C.sha256(a.repaired)[:12],
                           cold=dict(n_sobol=2 ** 14, top_k=16, seed=11), margin_km=MARGIN, tol=TOL),
               power_at_start=pw, cold_s=cold_s, n_checked=len(rows), n_orig_suboptimal=len(sub),
               n_orig_suboptimal_fail=sum(not r["passed"] for r in sub),
               n_any_fail=sum(not r["passed"] for r in rows), all_pass=all(r["passed"] for r in rows),
               windows=per_win, points=rows)
    od = C.out_dir("suboptcheck", res["config"])
    C.write_json(os.path.join(od, "check.json"), to_jsonable(res))
    md = [f"# Known sub-optimal labels vs cold solves (repaired: {os.path.basename(a.repaired)})", "",
          f"Checked {len(rows)} lattice points (stretches +-{MARGIN} km). Originally sub-optimal (> {TOL:g}): {len(sub)}. "
          f"Of those failing after repair: {res['n_orig_suboptimal_fail']}. Any point failing after repair: {res['n_any_fail']}.", "",
          "| n_X | stretch km | source | expected | found | in stretch | max orig deficit | fail after repair | window pts | window fails |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for p in per_win:
        md.append(f"| {p['n_X']:.3g} | {p['stretch'][0]}-{p['stretch'][1]} | {p['source']} | {p['expected']} | {p['found']} | "
                  f"{p['found_in_stretch']} | {p['max_orig_deficit']:.2e} | {p['n_fail_after_repair']} | "
                  f"{p['n_checked_in_window']} | {p['n_fail_any_point_in_window']} |")
    md += ["", "Per point (originally sub-optimal, plus any failing point):", "",
           "| n_X | L km | split | orig deficit | repaired vs cold | repaired flag | result |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["orig_suboptimal"] or not r["passed"]:
            md.append(f"| {r['n_X']:.3g} | {r['L']:.4f} | {r['split']} | {r['cold_gain_vs_orig']:.2e} | "
                      f"{r['cold_gain_vs_repaired']:.2e} | {r['repaired_flag']} | {'PASS' if r['passed'] else 'FAIL'} |")
    with open(os.path.join(od, "CHECK.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md[:18 + len(KNOWN)]), flush=True)
    print(f"wrote {os.path.relpath(od, ROOT)}", flush=True)
    sys.exit(0 if res["all_pass"] else 1)


if __name__ == "__main__":
    main()
