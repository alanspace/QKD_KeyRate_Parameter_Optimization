"""Task F: the nine figures, from generated results only, into results/figures/<git>_<cfg>/ + FIGURES.md.

Inputs (newest directory of each kind under results/, and the reference JSONs):
  benchmark_<preset>_*/{summary.json,new_warm_sweep.npz} for lim, lim_literal, legacy
  timing_lim_*/timing.json, physics_diff_*/physics_diff.json, gamma_report_*/gamma_report.json,
  dense_scan_*/dense_scan.json, Optimization/reference/reference_<preset>[_tail].json,
  legacy run JSONs (via Optimization/benchmark/legacy_runs.py, checksummed),
  Optimization/run_scripts/outputs/JAX/Comparison/qkd_horizon_comparison.md (thesis-claimed horizons).
Style: key rates in bits/pulse, log-y where R is shown, one colour per method in every figure (validated
categorical palette slots 1-6), presets in slots 7-8 + neutral grey, provenance stamp on every figure.

Usage: python Optimization/report/make_figures.py
"""
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "Optimization", "benchmark"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from src.qkd_opt import LEGACY, reparam  # noqa: E402
from src.qkd_opt.physics_lim import _gamma, theta_exact  # noqa: E402
from src.qkd_opt.provenance import stamp, stamp_figure, to_jsonable  # noqa: E402
import legacy_runs  # noqa: E402

NX = legacy_runs.NX
GRID = legacy_runs.GRID
INK = "#0b0b0b"
M_COL = {"new_warm": "#2a78d6", "new_cold": "#eb6834", "lookup_table": "#1baf7a",
         "legacy_DA": "#eda100", "legacy_Hessian": "#e87ba4", "legacy_Hessian_FoR": "#008300"}
M_LS = {"new_warm": "-", "new_cold": "-", "lookup_table": ":", "legacy_DA": "--", "legacy_Hessian": "-.",
        "legacy_Hessian_FoR": (0, (5, 1, 1, 1))}
M_LAB = {"new_warm": "new optimizer, warm sweep", "new_cold": "new optimizer, cold start",
         "lookup_table": "lookup table", "legacy_DA": "old DA run", "legacy_Hessian": "old Hessian run",
         "legacy_Hessian_FoR": "old Hessian_FoR run"}
P_COL = {"lim": "#6250d6", "lim_literal": "#e34948", "legacy": "#52514e"}
P_LS = {"lim": "-", "lim_literal": "--", "legacy": "-."}
NXLAB = {nx: f"$n_X=10^{int(np.log10(nx))}$" for nx in NX}
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "0.9",
                     "grid.linewidth": 0.6, "font.size": 9, "legend.frameon": False})


def newest(pattern):
    c = sorted(glob.glob(os.path.join(ROOT, "results", pattern)), key=os.path.getmtime)
    if not c:
        raise FileNotFoundError(pattern)
    return c[-1]


def ref(preset, tail=False):
    return json.load(open(os.path.join(ROOT, "Optimization", "reference",
                                       f"reference_{preset}{'_tail' if tail else ''}.json")))


def ref_arrays(r):
    P = r["points"]
    return dict(L=np.array([p["L"] for p in P]), n=np.array([p["n_X"] for p in P]),
                R=np.array([p["R"] for p in P], float), l=np.array([p["l"] for p in P], float),
                g=np.array([p["grad_norm_theta"] for p in P], float),
                noise=np.array([bool(p.get("noise_floor", False)) for p in P]),
                params=np.array([p["params"] for p in P], float),
                cliff=np.array([bool(p.get("cliff_edge", False)) for p in P]))


def horizons(r):
    return {h["n_X"]: h.get("horizon_km") for h in r["horizons"]}


def main():
    S = {}
    for p in ("lim", "lim_literal", "legacy"):
        d = newest(f"benchmark_{p}_*")
        S[p] = dict(dir=d, summary=json.load(open(os.path.join(d, "summary.json"))),
                    sweep=np.load(os.path.join(d, "new_warm_sweep.npz"), allow_pickle=True))
    timing = json.load(open(os.path.join(newest("timing_lim_*"), "timing.json")))
    pdiff = json.load(open(os.path.join(newest("physics_diff_*"), "physics_diff.json")))
    gam = json.load(open(os.path.join(newest("gamma_report_*"), "gamma_report.json")))
    dscan = json.load(open(os.path.join(newest("dense_scan_*"), "dense_scan.json")))
    R = {p: ref(p) for p in ("lim", "lim_literal", "legacy")}
    RT = {p: ref(p, tail=True) for p in ("lim", "legacy")}
    legacy = {}
    for name in ("legacy_DA", "legacy_Hessian", "legacy_Hessian_FoR"):
        run = legacy_runs.load(name)
        Rr, lr = legacy_runs.reevaluate(run, LEGACY)
        legacy[name] = dict(R=Rr, params=run["params"])
    cfg = dict(script="make_figures.py", inputs={p: os.path.relpath(S[p]["dir"], ROOT) for p in S},
               references={p: R[p]["stamp"]["config_hash"] for p in R},
               timing=timing["stamp"]["config_hash"], physics_diff=pdiff["stamp"]["config_hash"],
               gamma_report=gam["stamp"]["config_hash"], dense_scan=dscan["stamp"]["config_hash"])
    st = stamp(cfg)
    out = os.path.join(ROOT, "results", "figures", f"{st['git_commit']}_{st['config_hash']}")
    os.makedirs(out, exist_ok=True)
    caps = []

    def save(fig, name, caption, readoff):
        stamp_figure(fig, st)
        fig.savefig(os.path.join(out, name), dpi=200)
        plt.close(fig)
        caps.append((name, caption, readoff))
        print(f"[fig] {name}", flush=True)

    H = {p: horizons(R[p]) for p in R}
    rl = ref_arrays(R["lim"])

    # ---------- 1. key rate vs L, 6 panels ----------
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True)
    sw = S["lim"]["sweep"]
    for i, (ax, nx) in enumerate(zip(axes.ravel(), NX)):
        Rw = sw["R"][i]
        ax.semilogy(sw["L"][Rw > 0], Rw[Rw > 0], color=M_COL["new_warm"], lw=2, label="lim: " + M_LAB["new_warm"])
        for name, d in legacy.items():
            Rr = d["R"][i]
            ax.semilogy(GRID[Rr > 0], Rr[Rr > 0], color=M_COL[name], lw=1, ls=M_LS[name],
                        label=f"legacy preset: {M_LAB[name]}")
        sel = (rl["n"] == nx) & (rl["R"] > 0)
        ax.semilogy(rl["L"][sel], rl["R"][sel], "o", ms=4.5, mfc="none", mec=INK, mew=1.1, label="lim: reference")
        for p in ("lim", "lim_literal", "legacy"):
            h = H[p].get(nx)
            if h is not None:
                ax.axvline(h, color=P_COL[p], ls=P_LS[p], lw=1.1, label=f"horizon, preset {p}")
        if nx == 1e4:
            ax.plot([135.0], [1e-9], marker="*", ms=13, color=INK, ls="none", label="Lim et al. 2014: ~135 km")
        ax.set_ylim(1e-10, 3e-2)
        ax.set_xlim(0, 205)
        ax.set_title(NXLAB[nx])
        if i % 3 == 0:
            ax.set_ylabel("secret key rate R (bits/pulse)")
        if i >= 3:
            ax.set_xlabel("fiber length L (km)")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=7.5)
    fig.suptitle("Optimised secret key rate vs fiber length (R > 0 shown)")
    fig.tight_layout(rect=(0, 0.11, 1, 0.97))
    save(fig, "fig1_keyrate.png",
         "Optimised key rate of the lim preset (warm sweep, reference points) and the stored legacy runs re-evaluated "
         "in the legacy preset, one panel per block size. Vertical lines mark each preset's reference horizon; the "
         "star marks the ~135 km reach Lim et al. state for n_X = 1e4.",
         f"lim reference horizon at n_X=1e4: {H['lim'][1e4]:.2f} km (paper ~135 km)")

    # ---------- 2. preset comparison ----------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, nx in zip(axes, (1e4, 1e9)):
        for p in ("lim", "lim_literal", "legacy"):
            sw_ = S[p]["sweep"]
            i = NX.index(nx)
            Rp = sw_["R"][i]
            ax.semilogy(sw_["L"][Rp > 0], Rp[Rp > 0], color=P_COL[p], ls=P_LS[p], lw=2, label=f"preset {p}")
        ax.set_title(NXLAB[nx])
        ax.set_xlabel("fiber length L (km)")
        ax.set_xlim(0, 205)
    axes[0].set_ylabel("secret key rate R (bits/pulse)")
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Physics presets compared (new optimizer, warm sweep)")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    r0 = {p: float(S[p]["sweep"]["R"][NX.index(1e9), 0]) for p in S}
    save(fig, "fig2_presets.png",
         "Optimised key rate under the three physics presets for n_X = 1e4 and 1e9. lim uses eta_sys in the "
         "misalignment term, lim_literal the printed eta_ch, legacy the earlier src/qkd physics.",
         "R(0, 1e9) bits/pulse: " + ", ".join(f"{p} {v:.3e}" for p, v in r0.items()))

    # ---------- 3. horizon vs log10 nX ----------
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    x = np.log10(NX)
    mk = {"printed": "o", "exact": "s", "max": "^"}
    for m, v in gam["modes"].items():
        y = [v["horizons"][f"{nx:.0e}"]["horizon_km"] for nx in NX]
        ax.plot(x, [np.nan if q is None else q for q in y], color=P_COL["lim"], marker=mk[m], ms=6,
                ls={"printed": "-", "exact": "--", "max": ":"}[m], lw=1.4, label=f"lim, gamma {m}")
    ax.plot(x, [H["legacy"][nx] for nx in NX], color=P_COL["legacy"], marker="D", ms=5, ls="-.", lw=1.4,
            label="legacy preset (reference)")
    md = open(os.path.join(ROOT, "Optimization", "run_scripts", "outputs", "JAX", "Comparison",
                           "qkd_horizon_comparison.md")).read()
    claimed = [float(m_.group(1)) for m_ in re.finditer(r"\|\s*\$10\^\{\d\}\$\s*\|\s*[\d.]+\s*\|\s*([\d.]+)", md)]
    if len(claimed) == 6:
        ax.plot(x, claimed, ls="none", marker="x", ms=8, mew=1.6, color=INK, label="thesis-claimed optimised horizon")
    ax.plot([4], [135.0], marker="*", ms=14, color=INK, ls="none", label="Lim et al. 2014: ~135 km")
    ax.set_xlabel("log10 n_X (block size)")
    ax.set_ylabel("horizon: largest L with l > 0 (km)")
    ax.set_title("Horizon vs block size")
    ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    hp = gam["modes"]["printed"]["horizons"]["1e+04"]["horizon_km"]
    he = gam["modes"]["exact"]["horizons"]["1e+04"]["horizon_km"]
    save(fig, "fig3_horizon.png",
         "Horizon (largest fiber length with positive key length at the optimum) vs block size for the lim preset "
         "with printed, exact (Fung-Ma-Chau) and max gamma, the legacy preset, the horizons claimed in the earlier "
         "thesis material, and the ~135 km stated by Lim et al. for n_X = 1e4.",
         f"n_X=1e4 horizon: lim printed {hp} km, lim exact {he} km, paper ~135 km")

    # ---------- 4. parameters vs L ----------
    names = ["mu_1", "mu_2", "P_mu_1", "P_mu_2", "q_x (P_X)"]
    rt = ref_arrays(RT["lim"])
    ev = S["lim"]["summary"]["sweep_events"]
    for nx in (1e4, 1e9):
        i = NX.index(nx)
        fig, axes = plt.subplots(5, 1, figsize=(8.5, 11), sharex=True)
        cl = np.concatenate([rl["L"][(rl["n"] == nx) & rl["cliff"]], rt["L"][(rt["n"] == nx) & rt["cliff"]]])
        bev = [e["crossover_between_km"] for e in ev if e["type"] == "backfill" and e["n_X"] == nx]
        djs = [(j["L_from"], j["L_to"]) for j in dscan["lim"].get(f"{nx:.0e}", {}).get("jumps", [])]
        hzn = H["lim"][nx]
        for c, ax in enumerate(axes):
            Rw = sw["R"][i]
            m_ = Rw > 0
            ax.plot(sw["L"][m_], sw["params"][i, m_, c], color=M_COL["new_warm"], lw=2, label="lim: new optimizer, warm sweep")
            for name in ("legacy_DA", "legacy_Hessian"):
                Rr = legacy[name]["R"][i]
                ax.plot(GRID[Rr > 0], legacy[name]["params"][i, Rr > 0, c], color=M_COL[name], ls=M_LS[name], lw=1,
                        label=f"legacy preset: {M_LAB[name]}")
            sel = (rl["n"] == nx) & (rl["R"] > 0)
            ax.plot(rl["L"][sel], rl["params"][sel, c], "o", ms=4, mfc="none", mec=INK, mew=1, label="lim: reference")
            selt = (rt["n"] == nx) & (rt["R"] > 0)
            ax.plot(rt["L"][selt], rt["params"][selt, c], ".", ms=3, color=INK, label="lim: tail reference")
            for L0 in cl:
                ax.axvspan(L0 - 0.25, L0 + 0.25, color="#eb6834", alpha=0.25, lw=0)
            for a, b in bev:
                ax.axvline(0.5 * (a + b), color=M_COL["new_warm"], ls=":", lw=1)
            for a, b in djs:
                ax.axvline(0.5 * (a + b), color=INK, ls=":", lw=0.8)
            ax.set_ylabel(names[c])
            last_leg = max([GRID[legacy[nm]["R"][i] > 0].max() for nm in ("legacy_DA", "legacy_Hessian")
                            if (legacy[nm]["R"][i] > 0).any()] + [hzn or 0])
            ax.set_xlim(0, min(last_leg + 3, 205))
        axes[0].plot([], [], color="#eb6834", alpha=0.5, lw=6, label="cliff_edge reference point")
        axes[0].plot([], [], color=M_COL["new_warm"], ls=":", label="branch switch (warm sweep)")
        axes[0].plot([], [], color=INK, ls=":", label="dense scan: >0.05 change per km (continuous at 0.1 km)")
        axes[0].legend(fontsize=7, ncol=2, loc="upper left", bbox_to_anchor=(0, 1.45))
        axes[-1].set_xlabel("fiber length L (km)")
        fig.suptitle(f"Optimised parameters vs fiber length, {NXLAB[nx]} (points with R > 0)")
        fig.tight_layout(rect=(0, 0.02, 1, 0.95))
        save(fig, f"fig4_params_nx1e{int(np.log10(nx))}.png",
             f"The five optimised parameters for {NXLAB[nx].strip('$')} under the lim preset (warm sweep, reference, "
             "tail reference) and the stored legacy DA/Hessian runs (legacy preset). Orange bands mark cliff_edge "
             "reference points, blue dotted lines branch switches logged by the warm sweep, black dotted lines 1 km pairs of "
             "the dense scan with a normalised parameter change > 0.05 (continuous drift when refined to 0.1 km).",
             f"branch switches (warm sweep) at n_X={nx:.0e}: " +
             (", ".join(f"{0.5 * (a + b):.2f} km" for a, b in bev) or "none"))

    # ---------- 5. tail diagnostics ----------
    fig, axes = plt.subplots(3, 6, figsize=(17, 8.2))
    for j, nx in enumerate(NX):
        hzn = H["lim"].get(nx)
        sel = (rt["n"] == nx)
        if hzn is None or not sel.any():
            for r_ in range(3):
                axes[r_, j].set_visible(False)
            continue
        L = rt["L"][sel]
        Rv, lv, gv, cv, nv = rt["R"][sel], rt["l"][sel], rt["g"][sel], rt["cliff"][sel], rt["noise"][sel]
        a0, a1, a2 = axes[0, j], axes[1, j], axes[2, j]
        pos = Rv > 0
        a0.semilogy(L[pos], Rv[pos], "-", color=INK, lw=1)
        a0.semilogy(L[pos & ~cv], Rv[pos & ~cv], "o", ms=3, color=INK)
        a0.semilogy(L[pos & cv], Rv[pos & cv], "o", ms=5, color="#eb6834")
        a1.plot(L, lv, "-", color=INK, lw=1)
        a1.plot(L[~cv], lv[~cv], "o", ms=3, color=INK)
        a1.plot(L[cv], lv[cv], "o", ms=5, color="#eb6834")
        a1.axhline(0, color="0.5", lw=0.8)
        a2.semilogy(L, np.maximum(gv, 1e-16), "o", ms=3, color=INK)
        a2.semilogy(L[cv], np.maximum(gv[cv], 1e-16), "o", ms=5, color="#eb6834")
        a2.semilogy(L[nv], np.maximum(gv[nv], 1e-16), "s", ms=5, color="#eda100")
        a2.axhline(1e-3, color="#e34948", ls="--", lw=1)
        a2.axhline(1e-6, color="#e34948", ls=":", lw=1)
        for a_ in (a0, a1, a2):
            a_.axvline(hzn, color=P_COL["lim"], lw=1, ls="-")
            a_.set_xlim(hzn - 15.5, hzn + 1.5)
        a0.set_title(NXLAB[nx])
        a2.set_xlabel("fiber length L (km)")
        if j == 0:
            a0.set_ylabel("R (bits/pulse), R > 0")
            a1.set_ylabel("key length l (bits)")
            a2.set_ylabel("certificate |grad| (scaled)")
    axes[0, 0].plot([], [], "o", color="#eb6834", label="cliff_edge point (grad > 1e-3)")
    axes[0, 0].plot([], [], "s", color="#eda100", label="noise_floor point (1e-6 < grad <= 1e-3)")
    axes[0, 0].plot([], [], "o", color=INK, label="regular point")
    axes[0, 0].plot([], [], color=P_COL["lim"], label="reference horizon")
    axes[0, 0].plot([], [], color="#e34948", ls="--", label="1e-3 (cliff_edge) threshold")
    axes[0, 0].plot([], [], color="#e34948", ls=":", label="1e-6 (noise_floor) threshold")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=6, fontsize=8,
               bbox_to_anchor=(0.5, 0.025))
    fig.suptitle("Tail diagnostics, preset lim: last 15 km before the horizon (dense tail reference, seeds 2 3)")
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    ncl = int(rt["cliff"].sum())
    save(fig, "fig5_tail.png",
         "Key rate (log), key length l (linear) and the certificate gradient norm in the last 15 km before the horizon "
         "for every block size, from the dense tail reference of the lim preset. Orange points are cliff_edge points "
         "(R > 0, gradient > 1e-3) at the eps_sec = kappa*l discontinuity / R -> 0 scaling, yellow squares noise_floor "
         "points (gradient 1e-6 .. 1e-3).",
         f"cliff_edge points in the lim tail reference: {ncl} of {int((rt['R'] > 0).sum())} with R > 0")

    # ---------- 6. regret ECDFs ----------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for ax, key, title in ((axes[0], "main_values", "bulk (21 lengths x 6 n_X)"),
                           (axes[1], "tail_values", "tail (horizon - 15 km .. + 1 km)")):
        for p, methods in (("lim", ("new_cold", "new_warm", "lookup_table")),
                           ("legacy", ("legacy_DA", "legacy_Hessian", "legacy_Hessian_FoR"))):
            summ = S[p]["summary"]
            vals = summ[key] if key == "main_values" else (summ.get("tail_regret") or {}).get("tail_values")
            if not vals:
                continue
            Rref = np.array(vals["R_ref"], float)
            cl = np.array(vals["cliff"], bool)
            sel = (Rref > 0) & ~cl
            for m in methods:
                if m not in vals:
                    continue
                rg = (Rref[sel] - np.array(vals[m], float)[sel]) / Rref[sel]
                rg = np.sort(np.clip(np.where(np.isfinite(rg), rg, 1.0), 1e-16, 1.0))
                ax.step(rg, np.arange(1, rg.size + 1) / rg.size, where="post", color=M_COL[m], ls=M_LS[m], lw=1.6,
                        label=f"{M_LAB[m]} ({p} preset)")
        for v in (1e-6, 1e-3):
            ax.axvline(v, color="0.45", ls="--", lw=0.8)
            ax.text(v * 1.3, 0.03, f"{v:g}", fontsize=7, color="0.35")
        ax.set_xscale("log")
        ax.set_xlim(1e-16, 1.5)
        ax.set_title(title)
        ax.set_xlabel("regret (R_ref - R)/R_ref, clipped to [1e-16, 1]")
    axes[0].set_ylabel("fraction of points (cliff_edge excluded)")
    h6, l6 = axes[1].get_legend_handles_labels()
    fig.legend(h6, l6, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle("Regret vs the trusted reference: empirical CDFs")
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    pw = S["lim"]["summary"]["regret"]["new_warm"]["all"]["pct_within_1e6"]
    save(fig, "fig6_regret_cdf.png",
         "Empirical cumulative distribution of the regret against the trusted reference, for the new optimizer and "
         "lookup table (lim preset) and the stored legacy runs (legacy preset), bulk and tail points; cliff_edge "
         "points are excluded. A curve reaching 1 left of the 1e-6 line means every point is within 1e-6 relative.",
         f"lim, new optimizer warm sweep, bulk points within 1e-6: {pw:.1f} %")

    # ---------- 7. ablation chain ----------
    ch = pdiff["chain"]
    Lc, Nc = np.array(pdiff["chain_L"]), np.array(pdiff["chain_nX"])
    stages = [c["stage"] for c in ch]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    w = 0.38
    for k, (nx, col) in enumerate(((1e4, M_COL["new_cold"]), (1e9, M_COL["new_warm"]))):
        j = int(np.where((Lc == 100.0) & (Nc == nx))[0][0])
        Rs = np.array([c["R"][j] for c in ch], float)
        ratio = [np.log10(Rs[s] / Rs[s - 1]) if (Rs[s] > 0 and Rs[s - 1] > 0) else np.nan for s in range(1, len(Rs))]
        xs = np.arange(1, len(Rs)) + (k - 0.5) * w
        axes[0].bar(xs, np.nan_to_num(ratio), width=w, color=col, label=NXLAB[nx])
        for xx, rr, s in zip(xs, ratio, range(1, len(Rs))):
            if not np.isfinite(rr):
                axes[0].text(xx, 0, "no key", rotation=90, ha="center", va="bottom", fontsize=7)
        hz = [c["horizons"][f"{nx:.0e}"] for c in ch]
        dh = [(hz[s] - hz[s - 1]) if (isinstance(hz[s], (int, float)) and isinstance(hz[s - 1], (int, float))
                                     and hz[s] is not None and hz[s - 1] is not None) else np.nan
              for s in range(1, len(hz))]
        axes[1].bar(xs, np.nan_to_num(dh), width=w, color=col, label=NXLAB[nx])
    short = [re.sub(r"^\d+ ", "", s)[:34] for s in stages[1:]]
    for ax, yl in zip(axes, ("log10 R(stage)/R(previous) at L = 100 km", "horizon change vs previous stage (km)")):
        ax.set_xticks(np.arange(1, len(stages)), [f"{i}: {s}" for i, s in enumerate(short, 1)], rotation=35,
                      ha="right", fontsize=7)
        ax.axhline(0, color="0.4", lw=0.8)
        ax.set_ylabel(yl)
        ax.legend(fontsize=8)
    fig.suptitle("Ablation chain legacy -> lim -> lim_literal (each stage re-optimised)")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    save(fig, "fig7_ablation.png",
         "Effect of each switch from the legacy physics to lim (stages 1-7; stage 1 is the expm1 evaluation, numerics "
         "only) and then to the printed eta_ch (stage 8, lim_literal), each stage re-optimised: change of log10 R at 100 km and change of the horizon. "
         "Bars labelled 'no key' have R <= 0 at that stage or the previous one.",
         "stage 2 (m_Z fix) log10 R ratio at 100 km, n_X=1e9: "
         f"{np.log10(ch[2]['R'][int(np.where((Lc == 100.0) & (Nc == 1e9))[0][0])] / ch[1]['R'][int(np.where((Lc == 100.0) & (Nc == 1e9))[0][0])]):+.3f}")

    # ---------- 8. timing ----------
    fig, ax = plt.subplots(figsize=(9, 4.4))
    rows = []
    da = [d["seconds"] for d in timing["legacy_DA"]]
    rows.append(("old DA algorithm (archived settings), per point", min(da), max(da), M_COL["legacy_DA"]))
    tc = S["lim"]["summary"]["timing"]
    rows.append(("new optimizer, cold start (batched), per point", tc["new_cold"]["steady_per_point_ms"] / 1e3, None, M_COL["new_cold"]))
    rows.append(("new optimizer, warm sweep (6 chains), per point", tc["new_warm"]["steady_per_point_ms"] / 1e3, None, M_COL["new_warm"]))
    w_ = timing["jax_warm_batch1"]
    rows.append(("warm-started solve, batch 1, device-resident", w_["steady_device_us"]["median"] / 1e6, None, M_COL["new_warm"]))
    rows.append(("warm-started solve, batch 1, end-to-end", w_["end_to_end_us"]["median"] / 1e6, None, M_COL["new_warm"]))
    tw = timing.get("jax_warm_batch1_tail") or []
    if tw:
        dv = [t_["steady_device_us"]["median"] / 1e6 for t_ in tw]
        rows.append((f"warm-started solve, batch 1, device, tail (5 km below horizon, {len(tw)} n_X)", min(dv), max(dv),
                     M_COL["new_warm"]))
    lt = timing["lookup_table_batch1"]
    rows.append(("lookup table, batch 1, device-resident", lt["steady_device_us"]["median"] / 1e6, None, M_COL["lookup_table"]))
    rows.append(("lookup table, batch 1, end-to-end", lt["end_to_end_us"]["median"] / 1e6, None, M_COL["lookup_table"]))
    for y, (lab, a, b, col) in enumerate(rows[::-1]):
        if b is not None:
            ax.plot([a, b], [y, y], color=col, lw=6, solid_capstyle="butt")
            ax.text(b * 1.4, y, f"{a:.2g}-{b:.2g} s", va="center", fontsize=7.5)
        else:
            ax.plot([a], [y], "o", ms=8, color=col)
            ax.text(a * 1.4, y, f"{a:.2g} s", va="center", fontsize=7.5)
    ax.set_yticks(range(len(rows)), [r_[0] for r_ in rows[::-1]], fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(1e-6, 1e2)
    ax.set_xlabel("wall time per point (s), Apple M2 Pro CPU, float64")
    ax.set_title("Measured time per point, preset lim")
    ax.text(0.99, 0.02, f"JIT compile, reported separately: cold {tc['new_cold']['compile_overhead_s']:.0f} s, "
                        f"warm sweep {tc['new_warm']['compile_overhead_s']:.0f} s", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7.5, color="0.3")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "fig8_timing.png",
         "Measured wall time per optimisation point for the lim preset: the archived dual-annealing settings, the new "
         "optimizer (batched cold start, warm sweep, warm-started batch-1 solve at 100 km and 5 km below each horizon) "
         "and a lookup table. Compile time is "
         "excluded from the bars and shown as a note; every value is a stopwatch measurement on this machine.",
         f"warm-started batch-1 solve, device-resident median: {w_['steady_device_us']['median']:.0f} us")

    # ---------- 9. gamma ----------
    ops = gam["operating_points"]
    eps_med = float(np.median([o["eps"] for o in ops])) if ops else 1e-10
    fig, ax = plt.subplots(figsize=(8.5, 5))
    cs = np.logspace(2, 10, 120)
    bcol = {0.02: "#2a78d6", 0.05: "#eb6834", 0.1: "#1baf7a"}
    for dval, ls in ((2e3, "-"), (1e4, "--")):
        for b, col in bcol.items():
            r_ = []
            for c in cs:
                te = float(theta_exact(jnp.asarray(eps_med / 21.0), jnp.asarray(b), jnp.asarray(c), jnp.asarray(dval)))
                gp = float(_gamma(eps_med, b, c, dval))
                r_.append(gp / te if te > 0 else np.nan)
            ax.plot(cs, r_, color=col, ls=ls, lw=1.6, label=f"b={b}, d={dval:.0e}")
    if ops:
        ax.scatter([o["c"] for o in ops], [o["ratio"] for o in ops], s=14, color=INK, zorder=3,
                   label="optimizer operating points (lim reference)")
    ax.set_xscale("log")
    ax.set_xlabel("c = s_Z1 (number of single-photon Z events)")
    ax.set_ylabel("printed gamma / exact theta")
    ax.set_title(f"Lim's closed-form gamma vs the exact Fung-Ma-Chau theta (eps_sec = {eps_med:.2g}, a = eps_sec/21)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    rat = [o["ratio"] for o in ops if o["ratio"]]
    save(fig, "fig9_gamma.png",
         "Ratio of the printed closed-form gamma to the exact Fung-Ma-Chau deviation theta (same failure probability "
         "a = eps_sec/21) as a function of the Z-basis single-photon count c, for two X-basis counts d and three "
         "error ratios b; black points are the values at the optimizer's operating points. A ratio above 1 means the "
         "printed gamma is the more conservative (larger) correction.",
         f"median ratio at operating points: {float(np.median(rat)):.3f}" if rat else "no operating points")

    lines = [f"FIGURES (generated by Optimization/report/make_figures.py), git {st['git_commit']}, config {st['config_hash']}", ""]
    for name, cap, ro in caps:
        lines += [f"{name}", f"  {cap}", f"  Read off: {ro}", ""]
    open(os.path.join(out, "FIGURES.md"), "w").write("\n".join(lines))
    json.dump(to_jsonable(dict(stamp=st, config=cfg, figures=[c[0] for c in caps])),
              open(os.path.join(out, "figures.json"), "w"), indent=1)
    print(f"[fig] wrote {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
