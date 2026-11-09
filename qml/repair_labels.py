"""Amendment 4: repair the labels after the Phase 4 run (main .venv; refuses unless on AC).

1. Backward warm sweep: optimize_sweep on the same grid with L descending, same n_X, no horizon refinement.
2. Per point, keep the better of the forward label and the backward solution (R where either is positive, otherwise
   l/n_X; the backward one must win by > 1e-12 relative). This is build_reference.py's cont_up / cont_down rule.
3. Recompute the cliff_edge / noise_floor flags (same rules) and the scored mask. Splits and inputs are unchanged.
4. Crossover windows: +-1 km around every isolated parameter jump in the repaired labels (max|du| > 10x both
   neighbouring steps and > 1e-3), united with the below-horizon switches logged by either sweep.
5. Gate: cold solves (2^14 Sobol, top-16, seed 11) at every lattice point within +-3 km of every isolated jump, and at
   1,000 random scored points. Pass if no repaired label is worse than a cold solve by > 1e-9 relative. The
   reference-point gate (PREREGISTRATION section 3.4) is repeated.
Output: qml/data/repaired/labels_repaired_<git>_<cfg>.{npz,json} + SHA256SUMS (a subdirectory, so no step picks it
up automatically), and results/qml_repair_<git>_<cfg>/repair_summary.json.

Usage: .venv/bin/python qml/repair_labels.py [--quick]   (--quick: 2 n_X, 70-100 km; logic test, nothing saved to qml/data)
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
from qml.labels import crossovers as logged_crossovers  # noqa: E402


def isolated_jumps(d, mask):
    out = []
    for nx in np.unique(d["n_X"][mask]):
        m = np.where(mask & (d["n_X"] == nx) & d["on_lattice"] & (d["R"] > 0))[0]
        m = m[np.argsort(d["L"][m])]
        if m.size < 4:
            continue
        du = np.max(np.abs(np.diff(d["u"][m], axis=0)), axis=1)
        for k in range(1, len(du) - 1):
            if du[k] > 10 * max(du[k - 1], du[k + 1]) and du[k] > 1e-3:
                out.append(dict(n_X=float(nx), L_lo=float(d["L"][m][k]), L_hi=float(d["L"][m][k + 1]),
                                km=float(0.5 * (d["L"][m][k] + d["L"][m][k + 1])), du=float(du[k])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--labels", default=None)
    a = ap.parse_args()
    pw = C.power_state() if a.quick else C.require_ac("repair_labels")
    import jax
    jax.config.update("jax_enable_x64", True)
    from src.qkd_opt import get_preset, check_x64, reparam
    from src.qkd_opt.optimize import optimize_sweep
    from qml.cold import optimize_sliced
    from src.qkd_opt.provenance import stamp, to_jsonable
    check_x64()
    pre = get_preset("lim")
    src = a.labels or sorted(glob.glob(os.path.join(C.DATA, "labels_*.npz")))[0]
    d0 = dict(np.load(src, allow_pickle=False))
    meta0 = json.load(open(src.replace(".npz", ".json")))
    nx_list = [float(x) for x in meta0["config"]["nX"]]
    G = np.unique(d0["L"])
    if a.quick:
        nx_list = [10 ** 7.5, 1e9]
        G = G[(G >= 70) & (G <= 100)]
    keep = np.isin(d0["n_X"], nx_list) & np.isin(d0["L"], G)
    d = {k: (v[keep] if isinstance(v, np.ndarray) and v.shape[:1] == d0["L"].shape else v) for k, v in d0.items()}
    cfg = dict(script="qml/repair_labels.py", source=os.path.relpath(src, ROOT), source_sha=C.sha256(src)[:12],
               quick=a.quick, nX=nx_list, n_grid=int(G.size), rule="best of forward/backward (>1e-12 rel)",
               gate=dict(cold_n_sobol=2 ** 14, top_k=16, seed=11, n_random=1000, tol=1e-9))
    st = stamp(cfg)

    # 1. backward sweep
    t0 = time.perf_counter()
    bw = optimize_sweep(G[::-1], nx_list, pre, refine_horizon=False)
    sweep_s = time.perf_counter() - t0
    P, nL = len(nx_list), G.size
    get = lambda k: np.asarray(bw[k])[:, ::-1].reshape(P * nL, *np.asarray(bw[k]).shape[2:])
    bL = np.tile(G, P); bN = np.repeat(np.asarray(nx_list), nL)
    pos = {(float(x), float(y)): i for i, (x, y) in enumerate(zip(d["n_X"], d["L"]))}
    order = np.array([pos[(float(x), float(y))] for x, y in zip(bN, bL)])      # backward row -> label index
    Rb, lb = get("R"), get("l")

    # 2. best of forward / backward
    Rf, lf = d["R"][order], d["l"][order]
    anypos = (Rf > 0) | (Rb > 0)
    sf = np.where(anypos, Rf, lf / bN)
    sb = np.where(anypos, Rb, lb / bN)
    take = np.isfinite(sb) & (sb > sf + 1e-12 * np.abs(sf))
    rep = d.copy()
    for k, bk in (("params", "params"), ("theta", "theta"), ("R", "R"), ("l", "l"), ("grad_norm_theta", "grad_norm_theta"),
                  ("hess_eig", "hess_eig"), ("solver_iters", "solver_iters")):
        arr = rep[k].copy()
        arr[order[take]] = get(bk)[take]
        rep[k] = arr
    rep["u"] = np.asarray(reparam.u_of_theta(rep["theta"]))
    stt = rep["status"].copy(); stt[order[take]] = np.array([str(s) for s in get("status")[take]]); rep["status"] = stt
    srcs = rep["source"].copy(); srcs[order[take]] = "backward"; rep["source"] = srcs
    repaired = np.zeros(len(rep["L"]), bool); repaired[order[take]] = True
    rel_gain = np.zeros(len(rep["L"])); rel_gain[order[take]] = ((sb - sf) / np.abs(sf))[take]
    rep["repaired"], rep["repair_rel_gain"] = repaired, rel_gain

    # 3. flags
    R, g = rep["R"], rep["grad_norm_theta"]
    rep["cliff_edge"] = (R > 0) & (g > C.CLIFF_GTOL)
    rep["noise_floor"] = (R > 0) & (g > C.NOISE_GTOL) & (g <= C.CLIFF_GTOL)
    is_ref = rep["split"] == "ref"
    rep["scored"] = np.where(is_ref, rep["scored"], (R > 0) & ~rep["cliff_edge"])

    # 4. crossover windows
    jumps = isolated_jumps(rep, np.ones(len(R), bool))
    logged = logged_crossovers(bw["events"], meta0["horizons"]) + [
        dict(n_X=c["n_X"], crossover_km=c["crossover_km"]) for c in meta0["crossovers"]]
    xo = [dict(n_X=j["n_X"], km=j["km"], source="jump") for j in jumps]
    for c in logged:
        if not any(abs(np.log10(c["n_X"]) - np.log10(x["n_X"])) < 1e-6 and abs(c["crossover_km"] - x["km"]) < 0.1 for x in xo):
            xo.append(dict(n_X=c["n_X"], km=c["crossover_km"], source="logged"))
    in_xo = np.zeros(len(R), bool)
    for c in xo:
        in_xo |= (np.abs(np.log10(rep["n_X"]) - np.log10(c["n_X"])) < 1e-6) & (np.abs(rep["L"] - c["km"]) <= C.CROSSOVER_HALF_KM)
    rep["in_crossover"] = in_xo

    # 5. gate
    rng = np.random.default_rng(0)
    win = np.zeros(len(R), bool)
    for j in jumps:
        win |= (np.abs(np.log10(rep["n_X"]) - np.log10(j["n_X"])) < 1e-6) & (np.abs(rep["L"] - j["km"]) <= 3) \
               & rep["on_lattice"] & (R > 0)
    sc = np.where(rep["scored"] & rep["on_lattice"])[0]
    rnd = rng.choice(sc, min(1000 if not a.quick else 50, sc.size), replace=False)
    chk = np.unique(np.concatenate([np.where(win)[0], rnd]))
    t0 = time.perf_counter()
    print(f"[repair] backward sweep {sweep_s:.0f}s; gate: {chk.size} points ({int(win.sum())} in jump windows, "
          f"{rnd.size} random), 2^14 Sobol", flush=True)
    cold = optimize_sliced(rep["L"][chk], rep["n_X"][chk], pre, n_sobol=2 ** 14, top_k=16, seed=11, tag="gate")
    gate_s = time.perf_counter() - t0
    rel = (cold["R"] - R[chk]) / np.abs(R[chk])
    worse = chk[rel > 1e-9]
    rel_orig = (cold["R"] - d["R"][chk]) / np.abs(d["R"][chk])
    ref_gate = dict(n=0, worse=0, min_rel=0.0)
    for i in np.where(is_ref & np.isfinite(rep["R_ref"]) & (rep["R_ref"] > 0) & ~rep["ref_cliff"])[0]:
        r_ = (R[i] - rep["R_ref"][i]) / rep["R_ref"][i]
        ref_gate["n"] += 1; ref_gate["min_rel"] = min(ref_gate["min_rel"], float(r_))
        ref_gate["worse"] += int(r_ < -1e-9)
    gate_pass = (worse.size == 0) and ref_gate["worse"] == 0

    by_split = {s: int((repaired & (rep["split"] == s)).sum()) for s in np.unique(rep["split"])}
    summary = dict(stamp=st, config=cfg, power_at_start=pw, backward_sweep_s=sweep_s, gate_s=gate_s,
                   n_repaired=int(repaired.sum()), repaired_by_split=by_split,
                   repair_rel_gain_max=float(rel_gain.max()), repaired_points=[
                       dict(n_X=float(rep["n_X"][i]), L=float(rep["L"][i]), split=str(rep["split"][i]),
                            rel_gain=float(rel_gain[i])) for i in np.where(repaired & (rel_gain > 1e-9))[0]],
                   isolated_jumps_after_repair=jumps, crossovers=xo, n_crossover_windows_scored=int((in_xo & rep["scored"]).sum()),
                   gate=dict(n_checked=int(chk.size), n_window=int(win.sum()), n_random=int(rnd.size),
                             worse_than_cold=[dict(n_X=float(rep["n_X"][i]), L=float(rep["L"][i]),
                                                   rel=float(rel[np.where(chk == i)[0][0]])) for i in worse],
                             max_cold_gain_repaired=float(rel.max()), max_cold_gain_original=float(rel_orig.max()),
                             n_original_worse=int((rel_orig > 1e-9).sum())),
                   ref_gate=ref_gate, gate_pass=gate_pass)
    od = C.out_dir("repair", cfg)
    C.write_json(os.path.join(od, "repair_summary.json"), to_jsonable(summary))
    if not a.quick:
        rd = os.path.join(C.DATA, "repaired")
        os.makedirs(rd, exist_ok=True)
        f = os.path.join(rd, f"labels_repaired_{st['git_commit']}_{st['config_hash']}.npz")
        np.savez_compressed(f, **rep)
        meta = dict(meta0, repair=os.path.relpath(od, ROOT), crossovers_repaired=xo, gate_pass=gate_pass)
        C.write_json(f.replace(".npz", ".json"), to_jsonable(meta))
        with open(os.path.join(rd, "SHA256SUMS"), "a") as fh:
            for p_ in (f, f.replace(".npz", ".json")):
                fh.write(f"{C.sha256(p_)}  {os.path.basename(p_)}\n")
    print(json.dumps({k: summary[k] for k in ("backward_sweep_s", "n_repaired", "repaired_by_split", "repair_rel_gain_max",
                                               "n_crossover_windows_scored", "gate", "ref_gate", "gate_pass")},
                     indent=1, default=str), flush=True)
    print(f"crossovers after repair: {len(xo)}; wrote {os.path.relpath(od, ROOT)}", flush=True)
    if not gate_pass:
        sys.exit("repair_labels: GATE FAILED")


if __name__ == "__main__":
    main()
