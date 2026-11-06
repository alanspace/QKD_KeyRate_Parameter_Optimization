"""Certification of surrogate predictions (main .venv). PREREGISTRATION.md sections 1 and 7.

Every R reported anywhere in the study comes from here: the float64 lim model (src/qkd_opt/objective.evaluate,
eps_sec = kappa*l greatest fixed point) evaluated at the predicted parameters p = reparam.params_of_u(u_hat).
Model-predicted R is never used.

  certify(idx, u_hat)        R, l at the prediction; regret vs R_label (or R_ref on the ref split)
  hybrid(idx, u_hat, k)      optimize.local_batch from theta(u_hat) with max_iter = k (k = 1, 2, 3); mode 0 if
                             R(p_hat) > 0 else 1, scale = |value at the start| (as in optimize._solve);
                             regret, iterations used
  time_* functions           batch-1 wall time (median; device = block_until_ready, end-to-end = host arrays)

--pilot: 1000 stand-in predictions (labels of test points perturbed by N(0, 0.01) in u, clipped to the feasible
interval) to time one full certification pass + hybrid k = 1..3, plus batch-1 timing of the k-step solve, the cold
solve (optimize, 2^12 Sobol, top-8) and the warm solve from the nearest lower training node. Timing only.

Usage: .venv/bin/python qml/certify.py --pilot [--labels FILE]
       .venv/bin/python qml/certify.py --pred FILE [--labels FILE] [--hybrid]   (Phase 4)
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

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)
from src.qkd_opt import get_preset, check_x64, reparam  # noqa: E402
from src.qkd_opt import optimize as O  # noqa: E402

PRE = get_preset("lim")
MU3 = PRE.consts.mu3


def load_labels(path=None):
    path = path or sorted(glob.glob(os.path.join(C.DATA, "labels_*.npz")))[-1]
    return path, dict(np.load(path, allow_pickle=False))


def theta_of(u_hat):
    return reparam.theta_of_u(np.clip(u_hat, C.U_LO, 1 - C.U_LO))


def R_at(theta, L, nX):
    R, l = O.eval_starts(jnp.asarray(theta)[:, None, :], jnp.asarray(L), jnp.asarray(nX), PRE.consts, PRE.flags)
    return np.asarray(R)[:, 0], np.asarray(l)[:, 0]


def reference_R(d, idx):
    return np.where(d["split"][idx] == "ref", d["R_ref"][idx], d["R"][idx])


def certify(d, idx, u_hat):
    th = theta_of(u_hat)
    R, l = R_at(th, d["L"][idx], d["n_X"][idx])
    Rr = reference_R(d, idx)
    return dict(R=R, l=l, regret=(Rr - R) / Rr, u_err=np.max(np.abs(u_hat - d["u"][idx]), axis=1))


def _mode_scale(R, l, nX):
    mode = (R <= 0).astype(np.int64)
    return mode, O._scale_from(np.where(mode > 0, l / nX, R))


def hybrid(d, idx, u_hat, k):
    th = theta_of(u_hat)
    L, nX = d["L"][idx], d["n_X"][idx]
    R0, l0 = R_at(th, L, nX)
    mode, scale = _mode_scale(R0, l0, nX)
    thk, _, it, st = map(np.asarray, O.local_batch(jnp.asarray(th)[:, None, :], jnp.asarray(L), jnp.asarray(nX),
                                                   jnp.asarray(mode), jnp.asarray(scale), PRE.consts, PRE.flags,
                                                   max_iter=k))
    R, l = R_at(thk[:, 0], L, nX)
    Rr = reference_R(d, idx)
    return dict(R=R, regret=(Rr - R) / Rr, iters=it[:, 0], status=st[:, 0])


# ---------------------------------------------------------------- batch-1 timing
def _med(x):
    return float(np.median(x)) * 1e6


def time_kstep(d, idx, u_hat, k, reps=1):
    dev, e2e, iters = [], [], []
    for j in range(len(idx)):
        th = theta_of(u_hat[j:j + 1])
        L, nX = d["L"][idx[j:j + 1]], d["n_X"][idx[j:j + 1]]
        R0, l0 = R_at(th, L, nX)
        mode, scale = _mode_scale(R0, l0, nX)
        args = (jnp.asarray(th)[:, None, :], jnp.asarray(L), jnp.asarray(nX), jnp.asarray(mode), jnp.asarray(scale))
        if j == 0:
            jax.block_until_ready(O.local_batch(*args, PRE.consts, PRE.flags, max_iter=k))   # compile
        t = time.perf_counter_ns()
        out = jax.block_until_ready(O.local_batch(*args, PRE.consts, PRE.flags, max_iter=k))
        dev.append((time.perf_counter_ns() - t) / 1e9)
        t = time.perf_counter_ns()
        args_h = (jnp.asarray(th)[:, None, :], jnp.asarray(L), jnp.asarray(nX), jnp.asarray(mode), jnp.asarray(scale))
        out = O.local_batch(*args_h, PRE.consts, PRE.flags, max_iter=k)
        np.asarray(out[0])
        e2e.append((time.perf_counter_ns() - t) / 1e9)
        iters.append(int(np.asarray(out[2])[0, 0]))
    return dict(k=k, n=len(idx), device_us_median=_med(dev), end_to_end_us_median=_med(e2e),
                iters_median=float(np.median(iters)))


def time_cold(d, idx):
    O.optimize(float(d["L"][idx[0]]), float(d["n_X"][idx[0]]), PRE, n_sobol=4096, top_k=8)   # compile
    ts = []
    for i in idx:
        t = time.perf_counter()
        O.optimize(float(d["L"][i]), float(d["n_X"][i]), PRE, n_sobol=4096, top_k=8)
        ts.append(time.perf_counter() - t)
    return dict(n=len(idx), end_to_end_us_median=_med(ts), end_to_end_us_p90=float(np.percentile(ts, 90)) * 1e6,
                settings="optimize(), 2^12 Sobol, top-8 (benchmark cold settings)")


def neighbour_train(d, i):
    """Index of the nearest lower training node of the same n_X."""
    m = np.where((d["n_X"] == d["n_X"][i]) & (d["split"] == "train") & d["scored"] & (d["L"] < d["L"][i]))[0]
    return m[np.argmax(d["L"][m])] if len(m) else None


def time_warm(d, idx):
    ts, iters, dist = [], [], []
    for j, i in enumerate(idx):
        nb = neighbour_train(d, i)
        th = d["theta"][nb][None]
        if j == 0:
            O.warm_solve(th, d["L"][[i]], d["n_X"][[i]], PRE)
        t = time.perf_counter()
        w = O.warm_solve(th, d["L"][[i]], d["n_X"][[i]], PRE)
        ts.append(time.perf_counter() - t)
        iters.append(int(w["iters"][0]))
        dist.append(float(d["L"][i] - d["L"][nb]))
    return dict(n=len(idx), end_to_end_us_median=_med(ts), iters_median=float(np.median(iters)),
                neighbour_km_median=float(np.median(dist)), settings="optimize.warm_solve, max_iter 200")


def pilot(d, path):
    rng = np.random.default_rng(0)
    test = np.where((d["split"] == "test") & d["scored"])[0]
    tb = test[d["region"][test] == "bulk"]
    tt = test[d["region"][test] == "tail"]
    idx = np.sort(np.concatenate([rng.choice(tb, 700, replace=False), rng.choice(tt, 300, replace=False)]))
    tr = (d["split"] == "train") & d["scored"]
    lo, hi = C.u_range(d["u"][tr])
    u_hat = np.clip(d["u"][idx] + rng.normal(0, 0.01, (len(idx), 5)), lo, hi)
    cfg = dict(script="qml/certify.py --pilot", labels=os.path.basename(path), n=len(idx), noise_u=0.01)
    from src.qkd_opt.provenance import stamp, to_jsonable
    st = stamp(cfg)
    out = dict(stamp=st, config=cfg)
    t = time.perf_counter(); c = certify(d, idx, u_hat); out["certify_first_s"] = time.perf_counter() - t
    t = time.perf_counter(); c = certify(d, idx, u_hat); out["certify_s"] = time.perf_counter() - t
    out["pilot_regret_median"] = float(np.median(c["regret"]))
    out["hybrid"] = []
    for k in (1, 2, 3):
        t = time.perf_counter(); h = hybrid(d, idx, u_hat, k); first = time.perf_counter() - t
        t = time.perf_counter(); h = hybrid(d, idx, u_hat, k); s = time.perf_counter() - t
        out["hybrid"].append(dict(k=k, first_s=first, steady_s=s, regret_median=float(np.median(h["regret"])),
                                  iters_mean=float(np.mean(h["iters"]))))
        print(f"[certify pilot] hybrid k={k}: {s:.2f}s (first {first:.1f}s) regret median {np.median(h['regret']):.2e}",
              flush=True)
    sub = np.concatenate([idx[np.isin(idx, tb)][:35], idx[np.isin(idx, tt)][:15]])
    sub_u = u_hat[np.isin(idx, sub)]
    out["batch1_kstep"] = [time_kstep(d, sub, sub_u, k) for k in (1, 2, 3)]
    print("[certify pilot] batch-1 k-step:", out["batch1_kstep"], flush=True)
    out["batch1_cold"] = time_cold(d, sub[:20])
    print("[certify pilot] batch-1 cold:", out["batch1_cold"], flush=True)
    out["batch1_warm"] = time_warm(d, sub)
    print("[certify pilot] batch-1 warm:", out["batch1_warm"], flush=True)
    od = C.out_dir("pilot_certify", cfg)
    C.write_json(os.path.join(od, "pilot_certify.json"), to_jsonable(out))
    print(f"[certify pilot] wrote {os.path.relpath(od, ROOT)}", flush=True)


# ---------------------------------------------------------------- Phase 4
THRESH = (1e-6, 1e-4, 1e-3, 1e-2)


def metrics(r, u_err=None, iters=None):
    r = np.asarray(r, np.float64)
    if r.size == 0:
        return dict(n=0)
    out = dict(n=int(r.size), median=float(np.median(r)), p99=float(np.percentile(r, 99)), max=float(np.max(r)),
               **{f"pct_le_{t:g}": float(100 * np.mean(r <= t)) for t in THRESH},
               pct_no_key=float(100 * np.mean(r >= 1)))
    if u_err is not None:
        out.update(u_err_median=float(np.median(u_err)), u_err_max=float(np.max(u_err)))
    if iters is not None:
        out.update(iters_mean=float(np.mean(iters)))
    return out


def groups(d):
    """Named boolean masks over all label points (PREREGISTRATION section 4, Amendment 2, supplementary score)."""
    sc, xo = d["scored"], d["in_crossover"]
    agg = sc & ~xo
    g = {}
    for rg in ("bulk", "tail"):
        reg = d["region"] == rg
        g[f"test_{rg}"] = agg & (d["split"] == "test") & reg
        g[f"ref_{rg}"] = agg & (d["split"] == "ref") & reg
        for nx in C.HELDOUT_NX:
            g[f"heldout_{nx:.0e}_{rg}"] = agg & (d["split"] == "heldout") & (d["n_X"] == nx) & reg
        for nx in C.EXTRAP_NX:
            g[f"extrap_{nx:.0e}_{rg}"] = agg & (d["split"] == "extrap") & (d["n_X"] == nx) & reg
        # supplementary (owner request after the cliff_edge diagnosis): ref points INCLUDING the reference cliff_edge
        g[f"supp_ref_incl_cliff_{rg}"] = (d["split"] == "ref") & (d["R_ref"] > 0) & ~xo & reg
    ev = np.isin(d["split"], ["test", "ref", "heldout", "extrap"])
    g["crossover_windows"] = sc & xo & ev
    g["drift_window_listed"] = sc & d["in_drift"] & ev
    return g


def eval_union(d):
    g = groups(d)
    m = np.zeros(len(d["L"]), bool)
    for v in g.values():
        m |= v
    return m


def latest(pattern):
    fs = sorted(glob.glob(os.path.join(C.RESULTS, pattern)), key=os.path.getmtime)
    if not fs:
        sys.exit(f"no {pattern}")
    return fs[-1]


def cmd_select(d, path):
    from src.qkd_opt.provenance import stamp, to_jsonable
    f = latest("qml_tune_*/predictions_val.npz")
    P = dict(np.load(f))
    meta = json.load(open(os.path.join(os.path.dirname(f), "tune.json")))
    idx = P.pop("idx")
    keep = d["scored"][idx] & ~d["in_crossover"][idx]
    cfg = dict(script="qml/certify.py --select", tune=os.path.relpath(f, ROOT), labels=os.path.basename(path),
               criterion="median validation regret, bulk and tail pooled, crossover windows excluded; "
                         "ties broken by fewer parameters")
    rows = {}
    for key, u in P.items():
        c = certify(d, idx[keep], u[keep])
        fam, ci = key.split("__")
        npar = next(r["n_params"] for r in meta["runs"] if r["key"] == f"{fam}|{ci}")
        rows[key] = dict(family=fam, config_index=int(ci), n_params=npar, **metrics(c["regret"], c["u_err"]))
        print(f"[select] {key}: median {rows[key]['median']:.3e} within1e-6 {rows[key]['pct_le_1e-06']:.1f}%", flush=True)
    selected = {}
    for fam in ("PQC-full", "PQC-product", "NN", "RFF-ridge"):
        cand = [r for r in rows.values() if r["family"] == fam]
        best = min(cand, key=lambda r: (r["median"], r["n_params"]))
        selected[fam] = best["config_index"]
    od = C.out_dir("select", cfg)
    C.write_json(os.path.join(od, "selection.json"), to_jsonable(dict(stamp=stamp(cfg), config=cfg, val=rows,
                                                                        selected=selected)))
    print(f"[select] selected {selected}; wrote {os.path.relpath(od, ROOT)}", flush=True)


def neighbours(d, idx):
    """Scenario (ii) start: TRAIN n_X -> nearest lower dense training node; other n_X -> nearest lower lattice label."""
    nb = np.full(len(idx), -1)
    for nx in np.unique(d["n_X"][idx]):
        if np.isin(nx, [10.0 ** e for e in C.TRAIN_LOG10NX]):
            pool = np.where((d["n_X"] == nx) & (d["split"] == "train") & d["scored"])[0]
        else:
            pool = np.where((d["n_X"] == nx) & d["on_lattice"] & (d["R"] > 0))[0]
        pool = pool[np.argsort(d["L"][pool])]
        sel = np.where(d["n_X"][idx] == nx)[0]
        j = np.searchsorted(d["L"][pool], d["L"][idx[sel]], side="left") - 1
        nb[sel] = np.where(j >= 0, pool[np.clip(j, 0, None)], -1)
    return nb


def horizon_errors(d, R_all, idxq):
    out = {}
    for nx in np.unique(d["n_X"]):
        m = (d["n_X"][idxq] == nx) & d["on_lattice"][idxq]
        if not m.any():
            continue
        L, R = d["L"][idxq][m], R_all[m]
        h = float(d["horizon_label"][idxq][m][0])
        ok = np.isfinite(R)
        pos = L[ok & (R > 0)]
        out[f"{nx:.4g}"] = dict(h_label=h, h_pred=float(pos.max()) if pos.size else None,
                                err_km=float(pos.max() - h) if pos.size else None,
                                capped_at_grid_end=bool(pos.size and pos.max() >= L[ok].max()))
    return out


def cmd_final(d, path, skip_baselines=False):
    from src.qkd_opt.provenance import stamp, to_jsonable
    f = latest("qml_final_*/predictions_query.npz")
    P = dict(np.load(f))
    idxq = P.pop("idx")
    pos_in_q = {int(i): k for k, i in enumerate(idxq)}
    ev_all = eval_union(d)
    ev = np.where(ev_all)[0]
    smoke = int(os.environ.get("QML_SMOKE_EVAL", 0))   # smoke runs only: subsample evaluation points (recorded in cfg)
    if smoke:
        ev = np.sort(np.random.default_rng(0).choice(ev, smoke, replace=False))
    assert all(int(i) in pos_in_q for i in ev), "an evaluation point is missing from the query set"
    evq = np.array([pos_in_q[int(i)] for i in ev])
    G = groups(d)
    cfg = dict(script="qml/certify.py --final", predictions=os.path.relpath(f, ROOT), labels=os.path.basename(path),
               n_eval=int(ev.size), hybrid_k=[1, 2, 3], cold="optimize_batch 2^12 Sobol top-8 seed 7 (batched)",
               warm="warm_solve max_iter 200 from neighbours()", smoke_eval=smoke)
    C.require_ac("certify --final")
    res = dict(stamp=stamp(cfg), config=cfg, keys={}, baselines={}, timing={})
    perpoint = {"ev_idx": ev}
    for key, uq in P.items():
        var, model, seed = key.split("__")
        ok = np.all(np.isfinite(uq[evq]), axis=1)
        e_idx, u_e = ev[ok], uq[evq][ok]
        c = certify(d, e_idx, u_e)
        reg = np.full(ev.size, np.nan); reg[ok] = c["regret"]
        uer = np.full(ev.size, np.nan); uer[ok] = c["u_err"]
        hk = {}
        for k in (1, 2, 3):
            h = hybrid(d, e_idx, u_e, k)
            rk = np.full(ev.size, np.nan); rk[ok] = h["regret"]
            it = np.full(ev.size, np.nan); it[ok] = h["iters"]
            hk[k] = (rk, it)
            perpoint[f"{key}__k{k}"] = rk
        perpoint[f"{key}__k0"] = reg
        # horizon: R at every query lattice point
        qok = np.all(np.isfinite(uq), axis=1)
        Rq = np.full(len(idxq), np.nan)
        Rq[qok], _ = R_at(theta_of(uq[qok]), d["L"][idxq[qok]], d["n_X"][idxq[qok]])
        out = dict(variant=var, model=model, seed=int(seed[1:]), groups={}, horizon=horizon_errors(d, Rq, idxq))
        for gname, gm in G.items():
            sel = gm[ev]
            okg = sel & np.isfinite(reg)
            out["groups"][gname] = dict(k0=metrics(reg[okg], uer[okg]),
                                        **{f"k{k}": metrics(hk[k][0][okg], iters=hk[k][1][okg]) for k in (1, 2, 3)})
        res["keys"][key] = out
        g = out["groups"]
        print(f"[final] {key}: test_bulk median {g['test_bulk']['k0'].get('median', float('nan')):.2e} "
              f"tail {g['test_tail']['k0'].get('median', float('nan')):.2e}", flush=True)

    if not skip_baselines:
        # scenario (i): cold solve at every evaluation point (batched), regret vs the same references
        t = time.perf_counter()
        cold = O.optimize_batch(d["L"][ev], d["n_X"][ev], PRE, n_sobol=4096, top_k=8, seed=7)
        rc = (reference_R(d, ev) - cold["R"]) / reference_R(d, ev)
        res["baselines"]["cold_batched_s"] = time.perf_counter() - t
        perpoint["cold"] = rc
        # scenario (ii): warm solve from the neighbour
        nb = neighbours(d, ev)
        okn = nb >= 0
        t = time.perf_counter()
        w = O.warm_solve(d["theta"][nb[okn]], d["L"][ev[okn]], d["n_X"][ev[okn]], PRE)
        res["baselines"]["warm_batched_s"] = time.perf_counter() - t
        Rw, _ = R_at(w["theta"], d["L"][ev[okn]], d["n_X"][ev[okn]])
        rw = np.full(ev.size, np.nan); rw[okn] = (reference_R(d, ev[okn]) - Rw) / reference_R(d, ev[okn])
        itw = np.full(ev.size, np.nan); itw[okn] = w["iters"]
        dist = np.full(ev.size, np.nan); dist[okn] = d["L"][ev[okn]] - d["L"][nb[okn]]
        perpoint["warm"] = rw
        perpoint["warm_neighbour_km"] = dist
        for name, arr, it in (("cold", rc, None), ("warm", rw, itw)):
            res["baselines"][name] = {gname: metrics(arr[gm[ev] & np.isfinite(arr)],
                                                     iters=None if it is None else it[gm[ev] & np.isfinite(arr)])
                                      for gname, gm in G.items()}
        res["baselines"]["warm_neighbour_km"] = {
            rg: dict(median=float(np.nanmedian(dist[(d["region"][ev] == rg)])),
                     values=sorted(set(np.round(dist[(d["region"][ev] == rg) & np.isfinite(dist)], 5).tolist()))[:12])
            for rg in ("bulk", "tail")}
        # batch-1 timing: k-step from real surrogate predictions (dense PQC-full seed 0), cold, warm
        rng = np.random.default_rng(1)
        tb = np.where(G["test_bulk"][ev])[0]
        tt = np.where(G["test_tail"][ev])[0]
        pick = np.concatenate([rng.choice(tb, min(35, tb.size), replace=False), rng.choice(tt, min(15, tt.size), replace=False)])
        u_pick = P["dense__PQC-full__s0"][evq[pick]]
        res["timing"]["batch1_kstep"] = [time_kstep(d, ev[pick], u_pick, k) for k in (1, 2, 3)]
        res["timing"]["batch1_cold"] = time_cold(d, ev[pick[:20]])
        res["timing"]["batch1_warm"] = time_warm(d, ev[pick])
        print("[final] timing", res["timing"], flush=True)

    od = C.out_dir("certify_final", cfg)
    np.savez_compressed(os.path.join(od, "perpoint.npz"), **perpoint)
    C.write_json(os.path.join(od, "metrics.json"), to_jsonable(res))
    print(f"[final] wrote {os.path.relpath(od, ROOT)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=None)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--skip-baselines", action="store_true")
    a = ap.parse_args()
    check_x64()
    path, d = load_labels(a.labels)
    if a.pilot:
        C.require_ac("certify --pilot")
        pilot(d, path)
    elif a.select:
        cmd_select(d, path)
    elif a.final:
        cmd_final(d, path, a.skip_baselines)
    else:
        sys.exit("choose --pilot, --select or --final")


if __name__ == "__main__":
    main()
