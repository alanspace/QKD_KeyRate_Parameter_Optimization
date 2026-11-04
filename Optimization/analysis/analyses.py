"""Re-done analyses (item E), physics-based, every number produced here.

1. Sensitivity: at selected (L, n_X), perturb one parameter in logit space (theta_i +/- d,
   d in {0.1, 0.25, 0.5, 1.0}; both signs) and report regret = 1 - R(p')/R* with
   (a) the other four parameters frozen, (b) the other four re-optimised.
2. Robustness to attenuation drift: parameters optimal at alpha = 0.2 dB/km used at a drifted
   alpha'. regret = 1 - R(p_nom | alpha') / max_p R(p | alpha'). The old "retention"
   R(p_nom | alpha') / R(p_nom | alpha) is also printed to show it is mostly channel loss.
3. Extrapolation of a surrogate. The NN cannot be run here (torch not installed), so the
   surrogate is a lookup table of optimal parameters (linear in L and log10 n_X):
   (a) held-out n_X = 1e6 (table built from the other five n_X),
   (b) interior L (table nodes every 4 km, evaluated at the midpoints),
   (c) P_dc and e_mis transfer: nominal-optimal params used at changed P_dc / e_mis
       (the table has no P_dc / e_mis input), regret vs re-optimised.
   Regret is reported only where the re-optimised R > 0; points where the surrogate gives R <= 0
   while the optimum is positive count as regret 1.
(Roughness restricted to R > tau is in run_benchmark.py, all n_X labelled.)

Usage: python Optimization/analysis/analyses.py --preset legacy|lim_exact
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

from src.qkd_opt import get_preset, check_x64, reparam, with_consts  # noqa: E402
from src.qkd_opt.objective import evaluate  # noqa: E402
from src.qkd_opt.optimize import optimize_batch, optimize_sweep  # noqa: E402
from src.qkd_opt.provenance import stamp, stamp_figure, to_jsonable  # noqa: E402

DELTAS = [0.1, 0.25, 0.5, 1.0]
PNAMES = ["mu_1", "mu_2", "P_mu_1", "P_mu_2", "q_x"]


def R_of(p, L, nx, pre):
    return float(evaluate(jnp.asarray(p), L, nx, pre.consts, pre.flags)["R"])


def R_batch(P, L, nx, pre):
    f = jax.jit(jax.vmap(lambda p, l_, n: evaluate(p, l_, n, pre.consts, pre.flags)["R"]))
    return np.asarray(f(jnp.asarray(P), jnp.asarray(L, float), jnp.asarray(nx, float)))


def opt(L, nx, pre, n_sobol=16384, top_k=16, extra=None):
    return optimize_batch(np.atleast_1d(L), np.atleast_1d(nx), pre, n_sobol=n_sobol, top_k=top_k,
                          extra_starts=extra)


def reopt_others(theta, i, L, nx, pre, R_scale):
    mu3 = pre.consts.mu3

    def f(t4):
        th = jnp.concatenate([t4[:i], jnp.asarray([theta[i]]), t4[i:]])
        return -evaluate(reparam.params_of_theta(th, mu3), L, nx, pre.consts, pre.flags)["R"] / R_scale
    vg = jax.jit(jax.value_and_grad(f))
    x0 = np.delete(theta, i)
    res = minimize(lambda x: tuple(np.asarray(v, float) for v in vg(jnp.asarray(x))), x0, jac=True,
                   method="BFGS", options=dict(gtol=1e-10, maxiter=500))
    return -res.fun * R_scale


def sensitivity(pre):
    pts = [(50.0, 1e9), (100.0, 1e9), (150.0, 1e9), (100.0, 1e6), (100.0, 1e4)]
    rows = []
    for L, nx in pts:
        o = opt(L, nx, pre)
        Rs, th = o["R"][0], o["theta"][0]
        if Rs <= 0:
            rows.append(dict(L=L, n_X=nx, R_opt=Rs, note="no positive key"))
            continue
        for i in range(5):
            for d in DELTAS:
                for sgn in (+1, -1):
                    t2 = th.copy()
                    t2[i] += sgn * d
                    p2 = np.asarray(reparam.params_of_theta(jnp.asarray(t2), pre.consts.mu3))
                    Rf = R_of(p2, L, nx, pre)
                    Rr = reopt_others(t2, i, L, nx, pre, abs(Rs))
                    rows.append(dict(L=L, n_X=nx, R_opt=Rs, param=PNAMES[i], delta_logit=sgn * d,
                                     p_opt=float(o["params"][0][i]), p_perturbed=float(p2[i]),
                                     regret_frozen=1 - Rf / Rs, regret_reoptimised=1 - Rr / Rs))
        print(f"[sens] L={L} n_X={nx:.0e} done", flush=True)
    return rows


def robustness(pre):
    alphas = [0.18, 0.19, 0.195, 0.205, 0.21, 0.22]
    pts = [(50.0, 1e9), (100.0, 1e9), (150.0, 1e9), (100.0, 1e6), (100.0, 1e4)]
    rows = []
    for L, nx in pts:
        o = opt(L, nx, pre)
        p_nom, R_nom = o["params"][0], o["R"][0]
        for a in alphas:
            pa = with_consts(pre, alpha=a)
            oa = opt(L, nx, pa, extra=p_nom[None, None, :])
            R_star = oa["R"][0]
            R_used = R_of(p_nom, L, nx, pa)
            rows.append(dict(L=L, n_X=nx, alpha=a, R_nominal=R_nom, R_nominal_params_drifted=R_used,
                             R_reoptimised_drifted=R_star,
                             regret=(1 - R_used / R_star) if R_star > 0 else None,
                             old_retention_ratio=R_used / R_nom if R_nom > 0 else None))
        print(f"[robust] L={L} n_X={nx:.0e} done", flush=True)
    return rows


def _regret(R_true, R_sur):
    pos = R_true > 0
    r = np.where(R_sur[pos] > 0, (R_true[pos] - R_sur[pos]) / R_true[pos], 1.0)
    if r.size == 0:
        return dict(n=0)
    return dict(n=int(r.size), median=float(np.median(r)), p90=float(np.quantile(r, 0.9)),
                max=float(r.max()), n_surrogate_nonpositive=int(np.sum(R_sur[pos] <= 0)))


def extrapolation(pre):
    nxs = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
    Lg = np.arange(0.0, 200.0 + 1e-9, 2.0)
    sw = optimize_sweep(Lg, nxs, pre, global_every=5, refine_horizon=False)
    T, R = sw["params"], sw["R"]
    out = {}
    # (a) held-out n_X = 1e6 : linear in log10 n_X between 1e5 and 1e7
    P_ho = 0.5 * (T[1] + T[3])
    out["held_out_nX_1e6"] = _regret(R[2], R_batch(P_ho, Lg, np.full_like(Lg, 1e6), pre))
    # (b) interior L: nodes every 4 km (even indices), evaluate at odd indices
    res = {}
    for i, nx in enumerate(nxs):
        odd = np.arange(1, len(Lg) - 1, 2)
        P_mid = 0.5 * (T[i, odd - 1] + T[i, odd + 1])
        res[f"{nx:.0e}"] = _regret(R[i, odd], R_batch(P_mid, Lg[odd], np.full(odd.size, nx), pre))
    out["interior_L_4km_nodes"] = res
    # (c) P_dc and e_mis transfer at n_X = 1e9 and 1e6 (every 10 km)
    sel = np.arange(0, len(Lg), 5)
    for name, vals in (("p_dc", [3e-7, 1.2e-6, 3e-6]), ("e_mis", [pre.consts.e_mis * 0.5, pre.consts.e_mis * 2])):
        res = {}
        for v in vals:
            pv = with_consts(pre, **{name: v})
            for i in (2, 5):
                Ltrue = optimize_batch(Lg[sel], np.full(sel.size, nxs[i]), pv, n_sobol=4096, top_k=8,
                                       extra_starts=T[i, sel][:, None, :])
                Rsur = R_batch(T[i, sel], Lg[sel], np.full(sel.size, nxs[i]), pv)
                res[f"{name}={v:g}, n_X={nxs[i]:.0e}"] = _regret(Ltrue["R"], Rsur)
        out[f"transfer_{name}"] = res
        print(f"[extrap] {name} done", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True)
    a = ap.parse_args()
    pre = get_preset(a.preset)
    backend = check_x64()
    cfg = dict(script="analyses.py", preset=pre.name, deltas=DELTAS, flags=pre.flags._asdict(),
               consts=pre.consts._asdict())
    st = stamp(cfg)
    out_dir = os.path.join(ROOT, "results", f"analyses_{pre.name}_{st['git_commit']}_{st['config_hash']}")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.perf_counter()
    sens = sensitivity(pre)
    rob = robustness(pre)
    ext = extrapolation(pre)
    res = dict(stamp=st, config=cfg, backend=backend, sensitivity=sens, robustness=rob, extrapolation=ext,
               wall_time_s=time.perf_counter() - t0)
    with open(os.path.join(out_dir, "analyses.json"), "w") as fh:
        json.dump(to_jsonable(res), fh, indent=1)

    L = [f"# Analyses — preset `{pre.name}`", "",
         f"Generated by `Optimization/analysis/analyses.py`; git `{st['git_commit']}`, config `{st['config_hash']}`.", "",
         "## 1. Sensitivity (logit-space perturbation of one parameter)", "",
         "regret = 1 − R(p′)/R*. Columns: frozen others / re-optimised others. Δθ is the change of the logit "
         "coordinate of that parameter (see src/qkd_opt/reparam.py).", ""]
    for (Lk, nx) in sorted({(r["L"], r["n_X"]) for r in sens}):
        rs = [r for r in sens if r["L"] == Lk and r["n_X"] == nx and "param" in r]
        if not rs:
            L.append(f"- L={Lk} km, n_X={nx:.0e}: no positive key")
            continue
        L += [f"**L = {Lk} km, n_X = {nx:.0e}, R* = {rs[0]['R_opt']:.4e} bits/pulse**", "",
              "| param | " + " | ".join(f"Δθ={s * d:+g}" for d in DELTAS for s in (-1, 1)) + " |",
              "|---|" + "---|" * (2 * len(DELTAS))]
        for pn in PNAMES:
            cells = []
            for d in DELTAS:
                for s in (-1, 1):
                    r = next(x for x in rs if x["param"] == pn and x["delta_logit"] == s * d)
                    cells.append(f"{r['regret_frozen']:.2e} / {r['regret_reoptimised']:.2e}")
            L.append(f"| {pn} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## 2. Robustness to attenuation drift", "",
          "regret = 1 − R(p_nom | α′)/max_p R(p | α′). `old ratio` = R(p_nom | α′)/R(p_nom | α=0.2), the quantity "
          "behind the report's \"78.4 % retention\" (dominated by the channel loss itself).", "",
          "| L (km) | n_X | α′ (dB/km) | regret | old ratio | R* at α′ |", "|---|---|---|---|---|---|"]
    for r in rob:
        reg = "–" if r["regret"] is None else f"{r['regret']:.2e}"
        ratio = "–" if r["old_retention_ratio"] is None else f"{r['old_retention_ratio']:.3f}"
        L.append(f"| {r['L']} | {r['n_X']:.0e} | {r['alpha']} | {reg} | {ratio} | {r['R_reoptimised_drifted']:.3e} |")
    L += ["", "## 3. Extrapolation of a lookup-table surrogate (NN not runnable here)", "",
          "Regret statistics where the re-optimised R > 0 (surrogate R ≤ 0 counts as regret 1).", ""]
    for k, v in ext.items():
        if "n" in v:
            L.append(f"- **{k}**: {json.dumps(to_jsonable(v))}")
        else:
            L.append(f"- **{k}**:")
            for kk, vv in v.items():
                L.append(f"  - {kk}: {json.dumps(to_jsonable(vv))}")
    with open(os.path.join(out_dir, "ANALYSES.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")

    # figure: robustness regret vs alpha
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    marks = ["o", "s", "^", "D", "v"]
    for k, (Lk, nx) in enumerate(sorted({(r["L"], r["n_X"]) for r in rob})):
        rs = [r for r in rob if r["L"] == Lk and r["n_X"] == nx and r["regret"] is not None]
        ax.plot([r["alpha"] for r in rs], [max(r["regret"], 1e-12) for r in rs], marker=marks[k], ms=6, lw=1.5,
                color=colors[k], label=f"L={Lk:g} km, n_X={nx:.0e}")
    ax.set_yscale("log")
    ax.axvline(pre.consts.alpha, color="0.6", lw=0.8, ls="--")
    ax.set_xlabel("drifted attenuation α′ (dB/km); parameters optimised at α = 0.2")
    ax.set_ylabel("regret vs re-optimised (clipped at 1e-12)")
    ax.set_title(f"Robustness to attenuation drift, preset {pre.name}")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(True, color="0.9")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    stamp_figure(fig, st)
    fig.savefig(os.path.join(out_dir, "fig_robustness.png"), dpi=200)
    print(f"[analyses] wrote {os.path.relpath(out_dir, ROOT)}")


if __name__ == "__main__":
    main()
