"""Two-sided comparison of labels with cold solves at the Amendment 4 gate points (STATUS 12).

The repair gate (repair_labels.py) is one-sided: it fails a repaired label only if a cold solve (2^14 Sobol, top-16,
seed 11) beats it by > 1e-9 relative. This script rebuilds the gate's exact point set from the repair summary and the
repaired label file (windows of +-3 km around the isolated jumps after repair, plus the same 1,000 random scored points,
rng seed 0), repeats the cold solves, and counts both directions for the original and the repaired labels:
  cold better than label by > 1e-9  (label sub-optimal vs this cold solve)
  label better than cold by > 1e-9  (the cold solve missed a better optimum)
Usage: .venv/bin/python qml/check_cold_two_sided.py --repaired qml/data/repaired/labels_repaired_<...>.npz
Output: results/qml_coldcheck_<git>_<cfg>/two_sided.json
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

TOL = 1e-9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repaired", required=True)
    a = ap.parse_args()
    pw = C.require_ac("check_cold_two_sided")
    import jax
    jax.config.update("jax_enable_x64", True)
    from src.qkd_opt import get_preset, check_x64
    from src.qkd_opt.provenance import to_jsonable
    from qml.cold import optimize_sliced
    check_x64()
    meta = json.load(open(a.repaired.replace(".npz", ".json")))
    summ = json.load(open(os.path.join(ROOT, meta["repair"], "repair_summary.json")))
    rep = dict(np.load(a.repaired, allow_pickle=False))
    orig_path = os.path.join(ROOT, summ["config"]["source"])
    d0 = dict(np.load(orig_path, allow_pickle=False))
    assert np.array_equal(d0["L"], rep["L"]) and np.array_equal(d0["n_X"], rep["n_X"]), "row order differs"
    R = rep["R"]
    win = np.zeros(len(R), bool)
    for j in summ["isolated_jumps_after_repair"]:
        win |= (np.abs(np.log10(rep["n_X"]) - np.log10(j["n_X"])) < 1e-6) & (np.abs(rep["L"] - j["km"]) <= 3) \
               & rep["on_lattice"] & (R > 0)
    sc = np.where(rep["scored"] & rep["on_lattice"])[0]
    rnd = np.random.default_rng(0).choice(sc, min(1000, sc.size), replace=False)
    chk = np.unique(np.concatenate([np.where(win)[0], rnd]))
    assert chk.size == summ["gate"]["n_checked"], (chk.size, summ["gate"]["n_checked"])
    t0 = time.perf_counter()
    cold = optimize_sliced(rep["L"][chk], rep["n_X"][chk], get_preset("lim"), n_sobol=2 ** 14, top_k=16, seed=11, tag="two-sided")
    secs = time.perf_counter() - t0
    Rc = np.asarray(cold["R"])
    out = dict(config=dict(script="qml/check_cold_two_sided.py", repaired=os.path.relpath(a.repaired, ROOT),
                           original=summ["config"]["source"], cold=dict(n_sobol=2 ** 14, top_k=16, seed=11), tol=TOL),
               power_at_start=pw, cold_s=secs, n_points=int(chk.size), n_window=int(win.sum()), n_random=int(rnd.size))
    for name, lab in (("original", d0["R"]), ("repaired", R)):
        Rl = lab[chk]
        ok = Rl > 0
        rel = np.where(ok, (Rc - Rl) / np.where(ok, np.abs(Rl), 1.0), np.nan)
        cold_better = ok & (rel > TOL)
        label_better = ok & (rel < -TOL)
        pts = [dict(n_X=float(rep["n_X"][i]), L=float(rep["L"][i]), split=str(rep["split"][i]), rel=float(rel[k]))
               for k, i in enumerate(chk) if label_better[k]]
        out[name] = dict(n_with_key=int(ok.sum()), n_cold_better=int(cold_better.sum()),
                         max_cold_gain=float(np.nanmax(rel)), n_label_better=int(label_better.sum()),
                         max_label_gain=float(-np.nanmin(rel)), n_within_tol_two_sided=int((ok & (np.abs(rel) <= TOL)).sum()),
                         label_better_points=sorted(pts, key=lambda p: (p["n_X"], p["L"])))
        print(f"[two-sided] {name}: with key {ok.sum()}, cold better {cold_better.sum()} (max {np.nanmax(rel):.2e}), "
              f"label better {label_better.sum()} (max {-np.nanmin(rel):.2e}), within +-{TOL:g}: {(ok & (np.abs(rel) <= TOL)).sum()}", flush=True)
    od = C.out_dir("coldcheck", out["config"])
    C.write_json(os.path.join(od, "two_sided.json"), to_jsonable(out))
    print(f"wrote {os.path.relpath(od, ROOT)}", flush=True)


if __name__ == "__main__":
    main()
