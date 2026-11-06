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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=None)
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    check_x64()
    path, d = load_labels(a.labels)
    if a.pilot:
        C.require_ac("certify --pilot")
        pilot(d, path)
    else:
        sys.exit("Phase 4 mode not implemented yet")


if __name__ == "__main__":
    main()
