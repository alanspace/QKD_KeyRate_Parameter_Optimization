"""Phase 2: labels for the surrogate study (main .venv; PREREGISTRATION.md sections 2-4).

One optimize_sweep (preset lim, default sweep settings) over 15 block sizes on a shared grid G:
  the 1/32-km lattice (0.125 km steps below 100 km, 1/32 km from 100 to 200 km) plus every L value of the
  reference grids (reference_lim.json, reference_lim_tail.json), so labels and reference share points exactly.
Every label keeps params, theta, u, R, l, status, solver_iters, grad_norm_theta, hess_eig, branch_id, source.
Flags cliff_edge / noise_floor use the build_reference.py rules; inputs (v, u_L) and splits follow sections 2 and 4.
Quality gate (section 3.4): at every shared point with the reference, R_label >= R_ref (1 - tol_ref) unless the
reference flags it cliff_edge. Refuses to run unless on AC.

Usage: .venv/bin/python qml/labels.py [--quick]   (--quick: coarse grid, smoke test only, not saved to qml/data)
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from qml import common as C  # noqa: E402


def reference_points():
    out = {}
    for tag in ("main", "tail"):
        f = os.path.join(ROOT, "Optimization", "reference", f"reference_lim{'_tail' if tag == 'tail' else ''}.json")
        out[tag] = json.load(open(f))
    return out


def build_grid(refs, quick=False):
    if quick:
        k = np.arange(0, 6401, 32)                       # 1 km, smoke test
    else:
        k = np.concatenate([np.arange(0, 3200, 4), np.arange(3200, 6401)])
    lat = k / C.LATTICE
    ref_L = sorted({p["L"] for r in refs.values() for p in r["points"]})
    G = np.unique(np.concatenate([lat, np.asarray(ref_L, np.float64)]))
    return G, np.asarray(ref_L, np.float64)


def lattice_index(L):
    k = np.rint(L * C.LATTICE)
    on = np.abs(L * C.LATTICE - k) < 1e-9
    return np.where(on, k, -1).astype(np.int64), on


def crossovers(events, horizons):
    """Below-horizon branch switches with objective R -> crossover km (midpoint of the backfill bracket)."""
    hz = {round(h["n_X"]): h.get("horizon_km") for h in horizons}
    out = []
    for i, e in enumerate(events):
        if e["type"] != "branch_switch" or e.get("objective") != "R":
            continue
        bf = next((b for b in events[i + 1:] if b["type"] == "backfill" and b["n_X"] == e["n_X"]
                   and b["L_switch"] == e["L"]), None)
        km = float(np.mean(bf["crossover_between_km"])) if bf else float(e["L"])
        h = hz.get(round(e["n_X"]))
        if h is None or km < h:
            out.append(dict(n_X=e["n_X"], L_switch=e["L"], crossover_km=km, rel_gain=e["rel_gain"],
                            u_dist=e["u_dist"], bracket=bf["crossover_between_km"] if bf else None))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    pw = C.require_ac("labels") if not a.quick else C.power_state()

    import jax
    jax.config.update("jax_enable_x64", True)
    from src.qkd_opt import get_preset, check_x64
    from src.qkd_opt.optimize import optimize_sweep
    from src.qkd_opt import reparam
    from src.qkd_opt.provenance import stamp, to_jsonable
    from Optimization.reference import build_reference as BR
    assert (BR.CLIFF_GTOL, BR.NOISE_GTOL) == (C.CLIFF_GTOL, C.NOISE_GTOL)
    check_x64()
    pre = get_preset("lim")

    refs = reference_points()
    ref_h = {h["n_X"]: h["horizon_km"] for h in refs["main"]["horizons"]}
    tol_ref = {t: r["summary"]["tol_ref"] for t, r in refs.items()}
    G, ref_L = build_grid(refs, a.quick)
    nx_train = [10.0 ** e for e in C.TRAIN_LOG10NX]
    nx_list = nx_train + C.HELDOUT_NX + C.EXTRAP_NX
    cfg = dict(script="qml/labels.py", preset=pre.name, quick=a.quick, n_grid=int(G.size), grid_sha=C.sha256_bytes(G.tobytes()), nX=nx_list, sweep="optimize_sweep defaults",
               tol_ref=tol_ref, ref_horizons=ref_h)
    st = stamp(cfg)
    print(f"[labels] grid {G.size} points x {len(nx_list)} n_X = {G.size * len(nx_list)}; power {pw['source']}",
          flush=True)

    t0 = time.perf_counter()
    sw = optimize_sweep(G, nx_list, pre)
    wall = time.perf_counter() - t0
    print(f"[labels] sweep {wall:.1f} s; events {len(sw['events'])}", flush=True)

    P, nL = len(nx_list), G.size
    nX = np.repeat(np.asarray(nx_list), nL)
    L = np.tile(G, P)
    flat = lambda k: np.asarray(sw[k]).reshape(P * nL, *np.asarray(sw[k]).shape[2:])
    params, theta, R, l = flat("params"), flat("theta"), flat("R"), flat("l")
    g = flat("grad_norm_theta")
    status = np.array([str(s) for s in flat("status")])
    u = np.asarray(reparam.u_of_theta(theta))
    cliff = (R > 0) & (g > C.CLIFF_GTOL)
    noise = (R > 0) & (g > C.NOISE_GTOL) & (g <= C.CLIFF_GTOL)
    hz_list = [to_jsonable(h) for h in sw["horizon"]]
    h_lab = np.repeat([h.get("horizon_km") or np.nan for h in hz_list], nL)

    # inputs (section 2)
    Lh = C.horizon_fit(np.log10(list(ref_h.keys())), list(ref_h.values()))
    v = C.v_of_nx(nX)
    uL = L / Lh(np.log10(nX))

    # splits (section 4)
    k, on_lat = lattice_index(L)
    is_ref_L = np.isin(L, ref_L)
    ref_nx = np.isin(nX, list(ref_h.keys()))
    pos = (R > 0) & ~cliff
    tail = (L >= h_lab - C.TAIL_WINDOW_KM) & (L <= h_lab)
    region = np.where(tail, "tail", "bulk")
    is_train_nx = np.isin(nX, nx_train)
    split = np.full(P * nL, "none", dtype=object)
    m16, m4 = k % 16, k % 4
    tr = is_train_nx & on_lat
    split[tr & ~tail & (m16 == 0)] = "train"
    split[tr & ~tail & (m16 == 4)] = "val"
    split[tr & ~tail & np.isin(m16, [8, 12])] = "test"
    split[tr & tail & (m4 == 0)] = "train"
    split[tr & tail & (m4 == 1)] = "val"
    split[tr & tail & np.isin(m4, [2, 3])] = "test"
    # ref split: exactly the (n_X, L) pairs of the reference JSONs, scored against R_ref with the reference flags
    R_ref = np.full(P * nL, np.nan)
    ref_cliff = np.zeros(P * nL, bool)
    pos_of = {(float(a_), float(b_)): i for i, (a_, b_) in enumerate(zip(nX, L))}
    for r in refs.values():
        for p in r["points"]:
            i = pos_of.get((float(p["n_X"]), float(p["L"])))
            if i is not None:
                R_ref[i], ref_cliff[i] = p["R"], bool(p["cliff_edge"])
    is_ref_pt = np.isfinite(R_ref)
    # reference points that are also lattice split points (L = 0 km) keep their lattice split (section 4 train rule);
    # they are not scored as ref and are listed in the summary
    ref_on_lattice = is_ref_pt & on_lat & (split != "none")
    is_ref_pt &= ~ref_on_lattice
    split[is_ref_pt] = "ref"
    for nx in C.HELDOUT_NX:
        split[(nX == nx) & on_lat] = "heldout"
    for nx in C.EXTRAP_NX:
        split[(nX == nx) & on_lat] = "extrap"
    xo = crossovers(sw["events"], hz_list)
    in_xo = np.zeros(P * nL, bool)
    for c in xo:
        in_xo |= (nX == c["n_X"]) & (np.abs(L - c["crossover_km"]) <= C.CROSSOVER_HALF_KM)
    in_drift = (nX == C.DRIFT_WINDOW["n_X"]) & (L >= C.DRIFT_WINDOW["L_lo"]) & (L <= C.DRIFT_WINDOW["L_hi"])

    # quality gate (section 3.4)
    gate = dict(main=dict(better=0, equal=0, worse=0, worse_points=[], excluded_cliff=0, n=0),
                tail=dict(better=0, equal=0, worse=0, worse_points=[], excluded_cliff=0, n=0))
    idx = {(float(a_), float(b_)): i for i, (a_, b_) in enumerate(zip(nX, L))}
    for tag, r in refs.items():
        for p in r["points"]:
            if p["R"] <= 0:
                continue
            gt = gate[tag]
            if p["cliff_edge"]:
                gt["excluded_cliff"] += 1
                continue
            i = idx.get((float(p["n_X"]), float(p["L"])))
            if i is None:
                if not a.quick:
                    raise RuntimeError(f"reference point not in label grid: {p['n_X']} {p['L']}")
                continue
            gt["n"] += 1
            rel = (R[i] - p["R"]) / p["R"]
            if rel > 1e-12:
                gt["better"] += 1
            elif rel < -tol_ref[tag]:
                gt["worse"] += 1
                gt["worse_points"].append(dict(n_X=p["n_X"], L=p["L"], R_ref=p["R"], R_label=float(R[i]), rel=float(rel)))
            else:
                gt["equal"] += 1
            gt.setdefault("min_rel", 0.0)
            gt["min_rel"] = min(gt["min_rel"], float(rel))
    gate_pass = all(gt["worse"] == 0 for gt in gate.values())

    counts = {}
    ref_scored = is_ref_pt & (R_ref > 0) & ~ref_cliff
    for s in ("train", "val", "test", "ref", "heldout", "extrap"):
        for rg in ("bulk", "tail"):
            m = (split == s) & (region == rg) & (ref_scored if s == "ref" else pos)
            counts[f"{s}_{rg}"] = int(m.sum())
    counts["crossover_window_points_scored"] = int((in_xo & pos & (split != "none")).sum())
    counts["cliff_edge"] = int(cliff.sum())
    counts["noise_floor"] = int(noise.sum())
    ref_on_lattice_pts = [dict(n_X=float(nX[i]), L=float(L[i]), split=str(split[i])) for i in np.where(ref_on_lattice)[0]]
    summary = dict(stamp=st, config=cfg, sweep_wall_s=wall, ref_points_kept_in_lattice_split=ref_on_lattice_pts, n_points=int(P * nL), horizons=hz_list,
                   crossovers=xo, gate=gate, gate_pass=gate_pass, counts=counts,
                   cliff_edge_points=[dict(n_X=float(nX[i]), L=float(L[i]), R=float(R[i]), l=float(l[i]),
                                           grad=float(g[i])) for i in np.where(cliff)[0]],
                   status_counts={s_: int((status == s_).sum()) for s_ in sorted(set(status))},
                   events=sw["events"], power_at_start=pw)

    tag = "quick" if a.quick else f"{st['git_commit']}_{st['config_hash']}"
    od = C.out_dir("labels", cfg)
    C.write_json(os.path.join(od, "labels_summary.json"), to_jsonable(summary))
    arrays = dict(n_X=nX, L=L, v=v, u_L=uL, params=params, theta=theta, u=u, R=R, l=l,
                  grad_norm_theta=g, hess_eig=flat("hess_eig"), solver_iters=flat("solver_iters"),
                  branch_id=flat("branch_id"), status=status.astype("U64"),
                  source=np.array([str(s) for s in flat("source")]).astype("U32"),
                  cliff_edge=cliff, noise_floor=noise, horizon_label=h_lab, lattice_k=k, on_lattice=on_lat,
                  is_ref_L=is_ref_L, split=split.astype("U8"), region=region.astype("U4"), in_crossover=in_xo,
                  in_drift=in_drift, scored=np.where(is_ref_pt, ref_scored, pos), R_ref=R_ref, ref_cliff=ref_cliff)
    if a.quick:
        np.savez_compressed(os.path.join(od, "labels_quick.npz"), **arrays)
    else:
        os.makedirs(C.DATA, exist_ok=True)
        f = os.path.join(C.DATA, f"labels_{tag}.npz")
        np.savez_compressed(f, **arrays)
        C.write_json(os.path.join(C.DATA, f"labels_{tag}.json"), to_jsonable(
            {k_: summary[k_] for k_ in ("stamp", "config", "sweep_wall_s", "ref_points_kept_in_lattice_split", "n_points", "horizons", "crossovers",
                                         "gate", "gate_pass", "counts", "cliff_edge_points", "status_counts")}))
        with open(os.path.join(C.DATA, "SHA256SUMS"), "a") as fh:
            for p_ in (f, f.replace(".npz", ".json")):
                fh.write(f"{C.sha256(p_)}  {os.path.basename(p_)}\n")
    print(json.dumps(dict(gate_pass=gate_pass, gate={t_: {k_: v_ for k_, v_ in g_.items() if k_ != 'worse_points'}
                                                    for t_, g_ in gate.items()}, counts=counts,
                          crossovers=xo, horizons=[(h["n_X"], h.get("horizon_km")) for h in hz_list]),
                     indent=1, default=str), flush=True)
    if not gate_pass:
        sys.exit("labels: QUALITY GATE FAILED (see labels_summary.json gate.*.worse_points)")


if __name__ == "__main__":
    main()
