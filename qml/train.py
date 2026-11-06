"""Phase 4 training and prediction (.venv-qml; no physics import). PREREGISTRATION.md sections 5-6, Amendment 3.

  tune      6 configurations x {PQC-full, PQC-product, NN}, seed 0, E = 2000 full-batch Adam steps on the dense
            training set; RFF ridge on every PQC-full configuration's trained spectrum (lambda by validation MSE in
            z inside the fit). Writes predictions u_hat on the validation set.
  final     the selected configurations (results/qml_select_*/selection.json, written by certify.py --select),
            seeds 0-4, variants dense and sparse (Amendment 3): PQC-full, PQC-product, NN-tuned, NN-matched,
            RFF-ridge; plus lookup-2D and lookup-nX per variant. Writes predictions u_hat on the query set.
  latency   batch-1 device-resident and end-to-end medians and batch-1024 throughput of every dense seed-0 final
            model and both lookup tables (JAX, jitted, float64).

Data exchange with .venv is by files only: inputs from qml/data/labels_*.npz, outputs predictions_*.npz.
Usage: .venv-qml/bin/python qml/train.py tune|final|latency
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
from qml import models as M  # noqa: E402

E = int(os.environ.get("QML_E", 2000))                       # QML_E / QML_SEEDS: smoke runs only (recorded in cfg)
SEEDS = [int(x) for x in os.environ.get("QML_SEEDS", "0,1,2,3,4").split(",")]
LAMBDAS = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2]


def labels_path():
    return sorted(glob.glob(os.path.join(C.DATA, "labels_*.npz")))[-1]


def load():
    p = labels_path()
    return p, dict(np.load(p, allow_pickle=False))


def train_mask(d, variant):
    m = (d["split"] == "train") & d["scored"]
    if variant == "sparse":
        m &= d["lattice_k"] % 64 == 0
    return m


def query_mask(d):
    return (d["on_lattice"] & (d["L"] <= d["horizon_label"] + 1.0)) | (d["split"] == "ref")


def val_mask(d):
    return (d["split"] == "val") & d["scored"]


def family_maker(fam):
    if fam == "PQC-full":
        return lambda c: M.make_pqc(*c, True), M.PQC_GRID
    if fam == "PQC-product":
        return lambda c: M.make_pqc(*c, False), M.PQC_GRID
    if fam == "NN":
        return M.make_nn, M.NN_WIDTHS
    raise ValueError(fam)


def fit(fam, conf, seed, X, Z):
    maker, _ = family_maker(fam)
    init, apply, npar = maker(conf)
    w = init(seed)
    step, zeros, loss = M.make_trainer(apply, E)
    m, v = zeros(w), zeros(w)
    t0 = time.perf_counter()
    for i in range(E):
        w, m, v, l_ = step(w, m, v, i, X, Z)
    jax.block_until_ready(l_)
    return dict(w=w, apply=apply, n_params=M.n_params(w), train_s=time.perf_counter() - t0,
                final_loss=float(loss(w, X, Z)))


def ridge(a_scalings, conf, seed, X, Z, Xv, Zv):
    n, L_ = conf
    D2 = M.ridge_D2_for(5 * n * L_)
    W = M.sample_frequencies(a_scalings, D2, seed)
    F, Fv = M.rff_features(X, W), M.rff_features(Xv, W)
    best = None
    for lam in LAMBDAS:
        B = M.fit_ridge(F, Z, lam)
        mse = float(np.mean((np.clip(Fv @ B, -1, 1) - Zv) ** 2))
        if best is None or mse < best[0]:
            best = (mse, lam, B)
    return dict(W=W, B=best[2], lam=best[1], val_mse_z=best[0], n_params=int(best[2].size), D2=D2)


def ridge_predict(r, Xq):
    return np.clip(M.rff_features(Xq, r["W"]) @ r["B"], -1, 1)


def spectrum_summary(a):
    a = np.abs(np.asarray(a)).reshape(-1, 2)
    return dict(n_encoding_gates_per_input=int(a.shape[0]), max_frequency_uL=float(a[:, 0].sum()),
                max_frequency_v=float(a[:, 1].sum()), scalings_uL=a[:, 0].round(4).tolist(),
                scalings_v=a[:, 1].round(4).tolist(),
                note="spectrum = {sum c_g a_g, c_g in {-1,0,1}} per input; max = sum |a_g|")


def cmd_tune(a):
    path, d = load()
    tr, va = train_mask(d, "dense"), val_mask(d)
    lo, hi = C.u_range(d["u"][tr])
    X, Xv = M.xprime(d["u_L"][tr], d["v"][tr]), M.xprime(d["u_L"][va], d["v"][va])
    Z, Zv = jnp.asarray(C.u_to_z(d["u"][tr], lo, hi)), C.u_to_z(d["u"][va], lo, hi)
    cfg = dict(script="qml/train.py tune", labels=os.path.basename(path), E=E, seed=0, lambdas=LAMBDAS,
               families=["PQC-full", "PQC-product", "NN", "RFF-ridge"], n_train=int(tr.sum()), n_val=int(va.sum()))
    C.require_ac("train tune")
    st = C.stamp_qml(cfg)
    out, meta = {"idx": np.where(va)[0]}, dict(stamp=st, config=cfg, lo=lo.tolist(), hi=hi.tolist(), runs=[])
    pqc_full_w = {}
    for fam in ("PQC-full", "PQC-product", "NN"):
        _, grid = family_maker(fam)
        for ci, conf in enumerate(grid):
            r = fit(fam, conf, 0, X, Z)
            zv = np.asarray(jax.jit(r["apply"])(r["w"], Xv))
            key = f"{fam}|{ci}"
            out[key] = C.z_to_u(zv, lo, hi)
            meta["runs"].append(dict(key=key, family=fam, config=conf, config_index=ci, n_params=r["n_params"],
                                     train_s=r["train_s"], final_loss=r["final_loss"],
                                     val_mse_z=float(np.mean((zv - Zv) ** 2))))
            if fam == "PQC-full":
                pqc_full_w[ci] = r["w"]
            print(f"[tune] {key} {conf}: params {r['n_params']} train {r['train_s']:.0f}s loss {r['final_loss']:.3e}",
                  flush=True)
    for ci, conf in enumerate(M.PQC_GRID):
        r = ridge(np.asarray(pqc_full_w[ci]["a"]), conf, 0, np.asarray(X), np.asarray(Z), np.asarray(Xv), Zv)
        key = f"RFF-ridge|{ci}"
        out[key] = C.z_to_u(ridge_predict(r, np.asarray(Xv)), lo, hi)
        meta["runs"].append(dict(key=key, family="RFF-ridge", config=conf, config_index=ci, n_params=r["n_params"],
                                 lam=r["lam"], val_mse_z=r["val_mse_z"], D2=r["D2"],
                                 spectrum=spectrum_summary(pqc_full_w[ci]["a"])))
        print(f"[tune] {key}: params {r['n_params']} lambda {r['lam']:g} val mse {r['val_mse_z']:.3e}", flush=True)
    od = C.out_dir("tune", cfg)
    np.savez_compressed(os.path.join(od, "predictions_val.npz"), **{k.replace("|", "__"): v for k, v in out.items()})
    C.write_json(os.path.join(od, "tune.json"), meta)
    print(f"[tune] wrote {os.path.relpath(od, ROOT)}", flush=True)


def selection():
    f = sorted(glob.glob(os.path.join(C.RESULTS, "qml_select_*", "selection.json")), key=os.path.getmtime)[-1]
    return f, json.load(open(f))


PARTS = os.path.join(C.DATA, "final_parts")
MODELS = ["PQC-full", "PQC-product", "NN-tuned", "NN-matched", "RFF-ridge"]


def _device():
    return dict(backend=jax.default_backend(), devices=[str(x) for x in jax.devices()], jax=jax.__version__,
                x64=bool(jax.config.read("jax_enable_x64")))


def _part_path(var, name, s):
    return os.path.join(PARTS, f"{var}__{name}__s{s}.npz")


def _save_part(var, name, s, u_query, weights, meta):
    os.makedirs(PARTS, exist_ok=True)
    np.savez(_part_path(var, name, s), u_query=u_query, meta=json.dumps(meta),
             **{f"w__{k}": np.asarray(v) for k, v in weights.items()})


def _load_part(var, name, s):
    f = _part_path(var, name, s)
    if not os.path.exists(f):
        return None
    z = np.load(f)
    return dict(u_query=z["u_query"], meta=json.loads(str(z["meta"])),
                w={k[3:]: z[k] for k in z.files if k.startswith("w__")})


def _variant_data(d, var):
    tr, va = train_mask(d, var), val_mask(d)
    lo, hi = C.u_range(d["u"][tr])
    X = M.xprime(d["u_L"][tr], d["v"][tr])
    Z = jnp.asarray(C.u_to_z(d["u"][tr], lo, hi))
    Xv, Zv = np.asarray(M.xprime(d["u_L"][va], d["v"][va])), C.u_to_z(d["u"][va], lo, hi)
    return tr, lo, hi, X, Z, Xv, Zv


def _models(sel):
    return [("PQC-full", "PQC-full", sel["selected"]["PQC-full"]), ("PQC-product", "PQC-product", sel["selected"]["PQC-product"]),
            ("NN-tuned", "NN", sel["selected"]["NN"]), ("NN-matched", "NN", sel["selected"]["PQC-full"])]


def cmd_final(a):
    """Train the selected configurations; every (variant, model, seed) is checkpointed as a part as soon as it is
    finished and skipped when its part exists, so an interrupted run resumes where it stopped. --seeds / --variants
    restrict the work (one GPU job per seed); --assemble writes predictions_query.npz + final.json from the parts."""
    path, d = load()
    sel_path, sel = selection()
    q = query_mask(d)
    Xq = M.xprime(d["u_L"][q], d["v"][q])
    if a.assemble:
        return assemble(a, d, q, path, sel_path, sel)
    if a.import_cpu_s0:
        return import_cpu_s0(d, q, Xq, sel)
    C.require_ac("train final")
    seeds = a.seeds if a.seeds is not None else SEEDS
    for var in a.variants:
        tr, lo, hi, X, Z, Xv, Zv = _variant_data(d, var)
        for s in seeds:
            w_full = None
            for name, fam, ci in _models(sel):
                part = _load_part(var, name, s)
                if part is not None:
                    print(f"[final] {var} {name} s{s}: part exists, skipped", flush=True)
                    if name == "PQC-full":
                        w_full = part["w"]
                    continue
                conf = family_maker(fam)[1][ci]
                r = fit(fam, conf, s, X, Z)
                if name == "PQC-full":
                    w_full = r["w"]
                zq = np.asarray(jax.jit(r["apply"])(r["w"], Xq))
                _save_part(var, name, s, C.z_to_u(zq, lo, hi), r["w"],
                           dict(model=name, seed=s, variant=var, config=conf, config_index=ci, n_params=r["n_params"],
                                train_s=r["train_s"], final_loss=r["final_loss"], device=_device(), E=E))
                print(f"[final] {var} {name} s{s} {conf}: train {r['train_s']:.0f}s loss {r['final_loss']:.3e} "
                      f"on {jax.default_backend()}", flush=True)
            if _load_part(var, "RFF-ridge", s) is not None:
                print(f"[final] {var} RFF-ridge s{s}: part exists, skipped", flush=True)
                continue
            ci = sel["selected"]["RFF-ridge"]
            t0 = time.perf_counter()
            if ci == sel["selected"]["PQC-full"]:
                a_sc = np.asarray(w_full["a"])
            else:
                a_sc = np.asarray(fit("PQC-full", M.PQC_GRID[ci], s, X, Z)["w"]["a"])
            rr = ridge(a_sc, M.PQC_GRID[ci], s, np.asarray(X), np.asarray(Z), Xv, Zv)
            _save_part(var, "RFF-ridge", s, C.z_to_u(ridge_predict(rr, np.asarray(Xq)), lo, hi), dict(W=rr["W"], B=rr["B"]),
                       dict(model="RFF-ridge", seed=s, variant=var, config=M.PQC_GRID[ci], config_index=ci,
                            n_params=rr["n_params"], lam=rr["lam"], D2=rr["D2"], train_s=time.perf_counter() - t0,
                            device=_device(), E=E))
            print(f"[final] {var} RFF-ridge s{s}: lambda {rr['lam']:g}", flush=True)


def import_cpu_s0(d, q, Xq, sel):
    """Parts for dense seed 0 from the weights the stopped CPU run saved (qml/data/weights_*_dense_s0.npz):
    predictions are recomputed from the weights, nothing is retrained. Train times are copied from the run log."""
    log_s = {"PQC-full": 1196.0, "PQC-product": 224.0, "NN-tuned": 2.0, "NN-matched": 2.0}   # final_20261106_140206.log
    tr, lo, hi, X, Z, Xv, Zv = _variant_data(d, "dense")
    dev = dict(backend="cpu", devices=["Apple M2 Pro CPU (stopped run 2026-11-06 14:02-14:36)"], jax=jax.__version__, x64=True)
    for name, fam, ci in _models(sel):
        if _load_part("dense", name, 0) is not None:
            print(f"[import] dense {name} s0: part exists, skipped"); continue
        w = {k: jnp.asarray(v) for k, v in np.load(os.path.join(C.DATA, f"weights_{name}_dense_s0.npz")).items()}
        conf = family_maker(fam)[1][ci]
        _, apply, npar = family_maker(fam)[0](conf)
        zq = np.asarray(jax.jit(apply)(w, Xq))
        loss = float(jnp.mean((apply(w, X) - Z) ** 2))
        _save_part("dense", name, 0, C.z_to_u(zq, lo, hi), w,
                   dict(model=name, seed=0, variant="dense", config=conf, config_index=ci, n_params=npar,
                        train_s=log_s[name], final_loss=loss, device=dev, E=E, imported_from_cpu_run=True))
        print(f"[import] dense {name} s0: loss {loss:.3e} (log value for comparison in final_20261106_140206.log)")
    if _load_part("dense", "RFF-ridge", 0) is None:
        rw = np.load(os.path.join(C.DATA, "weights_RFF-ridge_dense_s0.npz"))
        ci = sel["selected"]["RFF-ridge"]
        zq = np.clip(M.rff_features(np.asarray(Xq), rw["W"]) @ rw["B"], -1, 1)
        _save_part("dense", "RFF-ridge", 0, C.z_to_u(zq, lo, hi), dict(W=rw["W"], B=rw["B"]),
                   dict(model="RFF-ridge", seed=0, variant="dense", config=M.PQC_GRID[ci], config_index=ci,
                        n_params=int(rw["B"].size), lam=None, D2=int(rw["W"].shape[0]), train_s=None, device=dev, E=E,
                        imported_from_cpu_run=True, note="lambda not saved by the stopped run"))
        print("[import] dense RFF-ridge s0")


def assemble(a, d, q, path, sel_path, sel):
    variants, seeds = a.variants, SEEDS
    missing = [f"{v}__{m}__s{s}" for v in variants for m in MODELS for s in seeds if _load_part(v, m, s) is None]
    if missing:
        sys.exit(f"assemble: {len(missing)} parts missing, e.g. {missing[:5]}")
    cfg = dict(script="qml/train.py final (assembled from parts)", labels=os.path.basename(path),
               selection=os.path.relpath(sel_path, ROOT), selected=sel["selected"], E=E, seeds=seeds, variants=variants,
               n_query=int(q.sum()))
    out, meta = {"idx": np.where(q)[0]}, dict(stamp=C.stamp_qml(cfg), config=cfg, variants={})
    for var in variants:
        tr, lo, hi, *_ = _variant_data(d, var)
        vm = dict(n_train=int(tr.sum()), lo=lo.tolist(), hi=hi.tolist(), runs=[])
        for s in seeds:
            for m in MODELS:
                part = _load_part(var, m, s)
                out[f"{var}__{m}__s{s}"] = part["u_query"]
                vm["runs"].append(part["meta"])
        u2, n2 = M.lookup_2d(d["v"][tr], d["u_L"][tr], d["u"][tr], d["v"][q], d["u_L"][q])
        un, nn_ = M.lookup_nx(d["n_X"][tr], d["L"][tr], d["u"][tr], d["n_X"][q], d["L"][q])
        out[f"{var}__lookup-2D__s0"] = np.clip(u2, lo, hi)
        out[f"{var}__lookup-nX__s0"] = np.where(np.isnan(un), np.nan, np.clip(un, lo, hi))
        vm["lookup_nodes"] = dict(lookup_2D=n2, lookup_nX=nn_)
        meta["variants"][var] = vm
    for m in ("PQC-full", "PQC-product", "NN-tuned", "NN-matched"):        # dense seed-0 weights for latency / IQM
        w = _load_part("dense", m, 0)["w"]
        np.savez(os.path.join(C.DATA, f"weights_{m}_dense_s0.npz"), **w)
    od = C.out_dir("final", cfg)
    np.savez_compressed(os.path.join(od, "predictions_query.npz"), **out)
    C.write_json(os.path.join(od, "final.json"), meta)
    print(f"[final] assembled {len(out) - 1} prediction sets; wrote {os.path.relpath(od, ROOT)}", flush=True)


# ---------------------------------------------------------------- latency
def _lat(fn, x1, xb, reps=300):
    f = jax.jit(fn)
    jax.block_until_ready(f(x1)); jax.block_until_ready(f(xb))
    dev = []
    for _ in range(reps):
        t = time.perf_counter_ns(); jax.block_until_ready(f(x1)); dev.append(time.perf_counter_ns() - t)
    e2e = []
    xh = np.asarray(x1)
    for _ in range(reps):
        t = time.perf_counter_ns(); np.asarray(f(jnp.asarray(xh))); e2e.append(time.perf_counter_ns() - t)
    tb = []
    for _ in range(20):
        t = time.perf_counter_ns(); jax.block_until_ready(f(xb)); tb.append(time.perf_counter_ns() - t)
    return dict(batch1_device_us=float(np.median(dev)) / 1e3, batch1_end_to_end_us=float(np.median(e2e)) / 1e3,
                batch1024_s=float(np.median(tb)) / 1e9, throughput_per_s=1024 / (float(np.median(tb)) / 1e9))


def _padded_rows(v, uL, u):
    rows = np.unique(v)
    n = max(int((v == r).sum()) for r in rows)
    XP, UP = np.zeros((len(rows), n)), np.zeros((len(rows), n, u.shape[1]))
    for i, r in enumerate(rows):
        m = v == r
        o = np.argsort(uL[m])
        x, y = uL[m][o], u[m][o]
        pad = n - len(x)
        XP[i] = np.concatenate([x, x[-1] + 1e-9 * np.arange(1, pad + 1)])
        UP[i] = np.concatenate([y, np.repeat(y[-1:], pad, 0)])
    return jnp.asarray(rows), jnp.asarray(XP), jnp.asarray(UP)


def cmd_latency(a):
    path, d = load()
    f = sorted(glob.glob(os.path.join(C.RESULTS, "qml_final_*", "final.json")), key=os.path.getmtime)[-1]
    fin = json.load(open(f))
    tr = train_mask(d, "dense")
    lo, hi = np.asarray(fin["variants"]["dense"]["lo"]), np.asarray(fin["variants"]["dense"]["hi"])
    cfg = dict(script="qml/train.py latency", final=os.path.relpath(f, ROOT), reps=300, batch=1024)
    C.require_ac("train latency")
    st = C.stamp_qml(cfg)
    rng = np.random.default_rng(0)
    q = np.where(query_mask(d))[0]
    xb_idx = rng.choice(q, 1024, replace=False)
    x1 = jnp.asarray([[0.6, 0.5]])           # (u_L, v) raw inputs; models apply x' inside
    xb = jnp.asarray(np.stack([d["u_L"][xb_idx], d["v"][xb_idx]], 1))
    loJ, hiJ = jnp.asarray(lo), jnp.asarray(hi)
    res = dict(stamp=st, config=cfg, models={})
    sel = fin["config"]["selected"]
    for name, fam, ci in (("PQC-full", "PQC-full", sel["PQC-full"]), ("PQC-product", "PQC-product", sel["PQC-product"]),
                          ("NN-tuned", "NN", sel["NN"]), ("NN-matched", "NN", sel["PQC-full"])):
        conf = family_maker(fam)[1][ci]
        init, apply, _ = family_maker(fam)[0](conf)
        w = {k: jnp.asarray(v) for k, v in np.load(os.path.join(C.DATA, f"weights_{name}_dense_s0.npz")).items()}
        fn = lambda x, w=w, apply=apply: loJ + (hiJ - loJ) * (1 + jnp.clip(apply(w, M.xprime(x[:, 0], x[:, 1])), -1, 1)) / 2
        res["models"][name] = dict(config=conf, backend="default.qubit (analytic), jitted" if fam.startswith("PQC")
                                   else "JAX MLP, jitted", circuits_per_prediction=1 if fam.startswith("PQC") else 0,
                                   shots=0 if fam.startswith("PQC") else None, **_lat(fn, x1, xb))
        print(f"[latency] {name}: {res['models'][name]}", flush=True)
    rw = np.load(os.path.join(C.DATA, "weights_RFF-ridge_dense_s0.npz"))
    W, B = jnp.asarray(rw["W"]), jnp.asarray(rw["B"])

    def ridge_fn(x):
        P = M.xprime(x[:, 0], x[:, 1]) @ W.T
        F = jnp.concatenate([jnp.ones((P.shape[0], 1)), jnp.cos(P), jnp.sin(P)], 1)
        return loJ + (hiJ - loJ) * (1 + jnp.clip(F @ B, -1, 1)) / 2
    res["models"]["RFF-ridge"] = dict(backend="JAX, jitted", **_lat(ridge_fn, x1, xb))
    rows, XP, UP = _padded_rows(d["v"][tr], d["u_L"][tr], d["u"][tr])

    def lk2(x):
        def one(uL, v):
            j = jnp.clip(jnp.searchsorted(rows, v, side="right") - 1, 0, rows.size - 2)
            w_ = jnp.clip((v - rows[j]) / (rows[j + 1] - rows[j]), 0.0, 1.0)
            ia = jax.vmap(lambda c: jnp.interp(uL, XP[j], UP[j, :, c]))(jnp.arange(5))
            ib = jax.vmap(lambda c: jnp.interp(uL, XP[j + 1], UP[j + 1, :, c]))(jnp.arange(5))
            return jnp.clip((1 - w_) * ia + w_ * ib, loJ, hiJ)
        return jax.vmap(one)(x[:, 0], x[:, 1])
    # equality check of the jitted lookup against the numpy implementation used for the predictions
    u_np, _ = M.lookup_2d(d["v"][tr], d["u_L"][tr], d["u"][tr], np.asarray(xb[:, 1]), np.asarray(xb[:, 0]))
    res["lookup_2D_jax_vs_numpy_max_abs"] = float(np.max(np.abs(np.asarray(jax.jit(lk2)(xb)) - np.clip(u_np, lo, hi))))
    res["models"]["lookup-2D"] = dict(backend="JAX, jitted (padded rows)", **_lat(lk2, x1, xb))
    print(f"[latency] lookup-2D: {res['models']['lookup-2D']} (jax vs numpy {res['lookup_2D_jax_vs_numpy_max_abs']:.1e})",
          flush=True)
    # shots a hardware run would need: standard error eps in each <Z_i>, S = (1 - z^2) / eps^2, median over query set
    fq = sorted(glob.glob(os.path.join(os.path.dirname(f), "predictions_query.npz")))[0]
    up = np.load(fq)["dense__PQC-full__s0"]
    z = C.u_to_z(up, lo, hi)
    res["pqc_shots_estimate"] = {f"eps_{e:g}": float(np.median(np.max((1 - z ** 2) / e ** 2, axis=1)))
                                 for e in (1e-2, 1e-3, 1e-4)}
    od = C.out_dir("latency", cfg)
    C.write_json(os.path.join(od, "latency.json"), res)
    print(f"[latency] wrote {os.path.relpath(od, ROOT)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tune", "final", "latency"])
    ap.add_argument("--variants", nargs="+", default=["dense", "sparse"])
    ap.add_argument("--seeds", nargs="+", type=int, default=None, help="final: only these seeds (one GPU job per seed)")
    ap.add_argument("--assemble", action="store_true", help="final: build predictions_query.npz from the parts")
    ap.add_argument("--import-cpu-s0", action="store_true", help="final: parts for dense seed 0 from the stopped CPU run")
    a = ap.parse_args()
    {"tune": cmd_tune, "final": cmd_final, "latency": cmd_latency}[a.cmd](a)


if __name__ == "__main__":
    main()
