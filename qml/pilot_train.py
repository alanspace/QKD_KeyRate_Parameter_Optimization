"""Phase 3 timing pilot, model side (.venv-qml; no physics import).

For every configuration of every trainable family (PREREGISTRATION.md section 5): compile time and steady time of
one full-batch Adam step (= one epoch), and the time to predict the full query set. Ridge: feature build + lambda
sweep + fit per configuration (frequencies from an *untrained* PQC, a = 1: timing only). Lookup tables: build +
predict. Nothing here is a result; the parameters are discarded.

Query set Q: every lattice point of every n_X with L <= h + 1 km, plus the reference points (section 7).

Usage: .venv-qml/bin/python qml/pilot_train.py [--labels qml/data/labels_*.npz]
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


def load(path):
    d = dict(np.load(path, allow_pickle=False))
    return d


def query_mask(d):
    return ((d["on_lattice"] & (d["L"] <= d["horizon_label"] + 1.0)) | (d["split"] == "ref"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=None)
    ap.add_argument("--steps", type=int, default=3, help="steady steps timed per configuration")
    a = ap.parse_args()
    path = a.labels or sorted(glob.glob(os.path.join(C.DATA, "labels_*.npz")))[-1]
    d = load(path)
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from qml import models as M

    tr = (d["split"] == "train") & d["scored"]
    lo, hi = C.u_range(d["u"][tr])
    X = M.xprime(d["u_L"][tr], d["v"][tr])
    Z = jnp.asarray(C.u_to_z(d["u"][tr], lo, hi))
    q = query_mask(d)
    XQ = M.xprime(d["u_L"][q], d["v"][q])
    cfg = dict(script="qml/pilot_train.py", labels=os.path.basename(path), labels_sha=C.sha256(path)[:12],
               n_train=int(tr.sum()), n_query=int(q.sum()), steps_timed=a.steps)
    st = C.stamp_qml(cfg)
    print(f"[pilot_train] n_train {int(tr.sum())} n_query {int(q.sum())}; jax {jax.__version__}", flush=True)

    res = dict(stamp=st, config=cfg, families={})
    fams = [("PQC-full", lambda c: M.make_pqc(*c, True), M.PQC_GRID),
            ("PQC-product", lambda c: M.make_pqc(*c, False), M.PQC_GRID),
            ("NN", M.make_nn, M.NN_WIDTHS)]
    for name, maker, grid in fams:
        rows = []
        for c in grid:
            init, apply, npar = maker(c)
            w = init(0)
            step, zeros, _ = M.make_trainer(apply, 2000)
            m, v = zeros(w), zeros(w)
            t = time.perf_counter()
            out = step(w, m, v, 0, X, Z); jax.block_until_ready(out[3]); compile_s = time.perf_counter() - t
            ts = []
            w, m, v = out[0], out[1], out[2]
            for i in range(a.steps):
                t = time.perf_counter()
                w, m, v, l_ = step(w, m, v, i + 1, X, Z); jax.block_until_ready(l_)
                ts.append(time.perf_counter() - t)
            pred = jax.jit(apply)
            _, pred_first, pred_s = M.timed(pred, w, XQ, reps=1)
            rows.append(dict(config=c, n_params=M.n_params(w), declared_params=npar, compile_s=compile_s,
                             step_s=float(np.median(ts)), predict_query_first_s=pred_first, predict_query_s=pred_s))
            print(f"[pilot_train] {name} {c}: params {M.n_params(w)} compile {compile_s:.1f}s "
                  f"step {np.median(ts):.3f}s predict {pred_s:.2f}s", flush=True)
        res["families"][name] = rows

    # ridge: timing only, frequencies from untrained scalings (a = 1)
    rows = []
    va = (d["split"] == "val") & d["scored"]
    Xv = np.asarray(M.xprime(d["u_L"][va], d["v"][va]))
    for (n, L_) in M.PQC_GRID:
        D2 = M.ridge_D2_for(5 * n * L_)
        t = time.perf_counter()
        W = M.sample_frequencies(np.ones((L_, n, 2)), D2, 0)
        F, Fv = M.rff_features(np.asarray(X), W), M.rff_features(Xv, W)
        for lam in [1e-10, 1e-8, 1e-6, 1e-4, 1e-2]:
            B = M.fit_ridge(F, np.asarray(Z), lam); _ = Fv @ B
        fit_s = time.perf_counter() - t
        t = time.perf_counter(); _ = M.rff_features(np.asarray(XQ), W) @ B; pq = time.perf_counter() - t
        rows.append(dict(config=(n, L_), D2=D2, n_params=5 * (2 * D2 + 1), fit_with_lambda_sweep_s=fit_s,
                         predict_query_s=pq))
    res["families"]["RFF-ridge"] = rows

    t = time.perf_counter()
    _, size2d = M.lookup_2d(d["v"][tr], d["u_L"][tr], d["u"][tr], d["v"][q], d["u_L"][q])
    lk2 = time.perf_counter() - t
    t = time.perf_counter()
    _, sizenx = M.lookup_nx(d["n_X"][tr], d["L"][tr], d["u"][tr], d["n_X"][q], d["L"][q])
    lkn = time.perf_counter() - t
    res["families"]["lookup"] = [dict(config="2D", nodes=size2d, build_and_predict_query_s=lk2),
                                 dict(config="per-nX", nodes=sizenx, build_and_predict_query_s=lkn)]

    od = C.out_dir("pilot_train", cfg)
    C.write_json(os.path.join(od, "pilot_train.json"), res)
    print(f"[pilot_train] wrote {os.path.relpath(od, ROOT)}", flush=True)


if __name__ == "__main__":
    main()
