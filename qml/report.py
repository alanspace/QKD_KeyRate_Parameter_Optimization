"""qml/RESULTS.md and figures from the Phase 4 result files (main .venv). No number is typed by hand.

Inputs (latest of each): results/qml_select_*/selection.json, qml_tune_*/tune.json, qml_final_*/final.json,
qml_latency_*/latency.json, qml/data/labels_*.json, qml/data/repaired/labels_repaired_*.{npz,json} (Amendment 4), and
two certifications chosen by their label file, not by date (certifications()):
  primary   = newest qml_certify_final_* against the repaired labels (groups and crossover windows from that file);
  secondary = newest qml_certify_final_* against the original labels.
STATUS 12 points (a sweep label beats the 2^14-Sobol cold solve) are read from results/qml_coldcheck_*/two_sided.json
and results/qml_suboptcheck_*/check.json plus the two points checked separately (listed in STATUS 12).
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from qml import common as C  # noqa: E402

MODELS = ["PQC-full", "PQC-product", "NN-tuned", "NN-matched", "RFF-ridge", "lookup-2D", "lookup-nX"]
TRAINED = MODELS[:5]
SHOWN_MODELS, SHOWN_TRAINED = list(MODELS), list(TRAINED)   # NN-matched dropped in main() if identical to NN-tuned
K = ["k0", "k1", "k2", "k3"]


def latest(p):
    fs = sorted(glob.glob(os.path.join(C.RESULTS, p)), key=os.path.getmtime)
    if not fs:
        sys.exit(f"missing {p}")
    return fs[-1]


def repaired_labels():
    fs = sorted(glob.glob(os.path.join(C.DATA, "repaired", "labels_repaired_*.npz")), key=os.path.getmtime)
    if not fs:
        sys.exit("missing qml/data/repaired/labels_repaired_*.npz (Amendment 4)")
    return fs[-1]


def certifications():
    """(primary, secondary) metrics.json paths: newest certification against the repaired labels, newest against the
    original labels. Chosen by config['labels'], never by date alone."""
    rep_name = os.path.basename(repaired_labels())
    by = {}
    for f in sorted(glob.glob(os.path.join(C.RESULTS, "qml_certify_final_*/metrics.json")), key=os.path.getmtime):
        lab_ = json.load(open(f))["config"]["labels"]
        by.setdefault(lab_, []).append(f)
    if rep_name not in by:
        sys.exit(f"no certify_final against {rep_name}")
    orig = [k for k in by if k.startswith("labels_") and not k.startswith("labels_repaired")]
    if len(orig) != 1:
        sys.exit(f"expected one original-label certification set, found {orig}")
    return by[rep_name][-1], by[orig[0]][-1]


def nn_identical(fin):
    """True if NN-tuned and NN-matched are the same configuration (then they are one result, reported once)."""
    cf = {m: {json.dumps(r["config"], sort_keys=True) for r in fin["variants"]["dense"]["runs"] if r["model"] == m}
          for m in ("NN-tuned", "NN-matched")}
    return cf["NN-tuned"] == cf["NN-matched"]


def fmt(x, sci=True):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:.2e}" if sci else f"{x:.1f}"


def ms(vals, sci=True):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return "–"
    if len(vals) == 1:
        return fmt(vals[0], sci)
    return f"{fmt(float(np.mean(vals)), sci)} ± {fmt(float(np.std(vals, ddof=1)), sci)}"


def collect(M, var, model, group, k="k0", field="median"):
    out = []
    for key, v in M["keys"].items():
        if v["variant"] == var and v["model"] == model:
            g = v["groups"].get(group, {}).get(k, {})
            if g.get("n", 0) > 0:
                out.append(g.get(field))
    return out


def table(M, var, groups, k="k0"):
    L = ["| model | split | n | median | p99 | max | ≤1e-6 % | ≤1e-4 % | ≤1e-3 % | ≤1e-2 % | no key % | u-err median |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for model in SHOWN_MODELS:
        for g in groups:
            n = collect(M, var, model, g, k, "n")
            if not n:
                continue
            row = [model, g, str(int(np.median(n)))]
            row += [ms(collect(M, var, model, g, k, f)) for f in ("median", "p99", "max")]
            row += [ms(collect(M, var, model, g, k, f), sci=False) for f in
                    ("pct_le_1e-06", "pct_le_0.0001", "pct_le_0.001", "pct_le_0.01", "pct_no_key")]
            row += [ms(collect(M, var, model, g, k, "u_err_median")) if k == "k0" else "–"]
            L.append("| " + " | ".join(row) + " |")
    return "\n".join(L)


def baseline_rows(M, groups):
    L = ["| baseline | split | n | median | max | ≤1e-6 % | no key % | iterations (mean) |", "|---|---|---|---|---|---|---|---|"]
    for name in ("cold", "warm"):
        for g in groups:
            b = M["baselines"][name][g]
            if b.get("n", 0) == 0:
                continue
            L.append(f"| {name} | {g} | {b['n']} | {fmt(b['median'])} | {fmt(b['max'])} | {fmt(b['pct_le_1e-06'], False)} | "
                     f"{fmt(b['pct_no_key'], False)} | {fmt(b.get('iters_mean'), False) if 'iters_mean' in b else '–'} |")
    return "\n".join(L)


def horizon_table(M, var):
    nxs = sorted({nx for v in M["keys"].values() for nx in v["horizon"]}, key=float)
    L = ["| model | " + " | ".join(nxs) + " |", "|---|" + "---|" * len(nxs)]
    for model in SHOWN_MODELS:
        row = [model]
        for nx in nxs:
            vals = [v["horizon"].get(nx, {}).get("err_km") for v in M["keys"].values()
                    if v["variant"] == var and v["model"] == model]
            row.append(ms([x for x in vals if x is not None], sci=False) if any(x is not None for x in vals) else "–")
        L.append("| " + " | ".join(row) + " |")
    return "\n".join(L)


def figures(M, P, d, od):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ev = P["ev_idx"]
    from qml.certify import groups
    G = groups(d)
    figs = []
    for var in ("dense", "sparse"):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
        for ax, g in zip(axes, ("test_bulk", "test_tail")):
            m = G[g][ev]
            for model, col in zip(SHOWN_MODELS + ["cold", "warm"], plt.cm.tab10.colors):
                keys = [model] if model in ("cold", "warm") else [k[:-4] for k in P if k.startswith(f"{var}__{model}__") and k.endswith("__k0")]
                for j, key in enumerate(sorted(keys)):
                    r = P[key if model in ("cold", "warm") else f"{key}__k0"][m]
                    r = np.sort(np.clip(r[np.isfinite(r)], 1e-16, 1))
                    if r.size == 0:
                        continue
                    ax.step(r, np.arange(1, r.size + 1) / r.size, color=col, lw=1.6 if j == 0 else 0.6,
                            alpha=1 if j == 0 else 0.5, label=model if j == 0 else None)
            ax.set_xscale("log"); ax.set_xlim(1e-16, 1.5); ax.axvline(1e-6, color="0.5", ls="--", lw=0.8)
            ax.set_title(f"{g} ({var} training)"); ax.set_xlabel("regret (R_ref - R(p_hat)) / R_ref, clipped to [1e-16, 1]")
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("fraction of points"); axes[1].legend(fontsize=7, loc="lower right")
        fig.suptitle("Regret CDF of surrogate predictions (no Newton steps); thin lines: seeds 1-4", fontsize=10)
        st = M["stamp"]
        fig.text(0.995, 0.005, f"git {st['git_commit']} | cfg {st['config_hash']} | {st['cpu']} | power {st['power']['source']}",
                 ha="right", fontsize=6, color="0.4")
        f = os.path.join(od, f"regret_cdf_{var}.png")
        fig.tight_layout(); fig.savefig(f, dpi=150); plt.close(fig)
        figs.append(f)
    return figs


def usefulness(M, lat, var="dense", T=None):
    """Post-hoc operationalisation of Section 7's criterion (stated in RESULTS.md); raw numbers are tabulated too.
    T: batch-1 timing block to use (default: M's own); regrets always come from M."""
    T = T or M["timing"]
    kstep = {f"k{t['k']}": t["end_to_end_us_median"] for t in T["batch1_kstep"]}
    cold_us, warm_us = T["batch1_cold"]["end_to_end_us_median"], T["batch1_warm"]["end_to_end_us_median"]
    lk = lat["models"]["lookup-2D"]["batch1_end_to_end_us"]
    rows = []
    for model in SHOWN_TRAINED + ["lookup-2D"]:
        s_us = lat["models"].get(model, {}).get("batch1_end_to_end_us")
        for g in ("test_bulk", "test_tail"):
            med = np.mean(collect(M, var, model, g, "k0", "median"))
            p6 = np.mean(collect(M, var, model, g, "k0", "pct_le_1e-06"))
            lk_med = np.mean(collect(M, var, "lookup-2D", g, "k0", "median"))
            lk_p6 = np.mean(collect(M, var, "lookup-2D", g, "k0", "pct_le_1e-06"))
            b = "n/a (is the floor)" if model == "lookup-2D" else (
                "yes" if (med < lk_med and p6 > lk_p6 and s_us is not None and s_us <= 10 * lk) else "no")
            out = {}
            for scen, ref_name, ref_us in (("i", "cold", cold_us), ("ii", "warm", warm_us)):
                ref_p6 = M["baselines"][ref_name][g]["pct_le_1e-06"]
                ok = []
                for k in ("k1", "k2", "k3"):
                    p6k = np.mean(collect(M, var, model, g, k, "pct_le_1e-06"))
                    tot = (s_us or np.nan) + kstep[k]
                    if p6k >= ref_p6 and tot < ref_us:
                        ok.append(f"{k} ({tot:.0f} µs, {p6k:.1f} %)")
                out[scen] = "yes: " + ", ".join(ok) if ok else "no"
            rows.append((model, g, fmt(med), fmt(p6, False), fmt(s_us, False), b, out["i"], out["ii"]))
    L = ["| model | split | median regret | ≤1e-6 % | batch-1 e2e µs | (b) beats lookup-2D at ≤10× its latency | "
         "(i) hybrid reaches cold-solve class faster | (ii) hybrid reaches warm-solve class faster |",
         "|---|---|---|---|---|---|---|---|"]
    L += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(L), dict(kstep=kstep, cold_us=cold_us, warm_us=warm_us, lookup_us=lk)


def status12_points(d, d0, P, P0):
    """The STATUS 12 points (a sweep label beats the 2^14-Sobol/top-16 cold solve) and how they touch the numbers."""
    tw = json.load(open(latest("qml_coldcheck_*/two_sided.json")))
    pts = {(round(np.log10(p["n_X"]), 6), round(p["L"], 6)): p for p in tw["repaired"]["label_better_points"]}
    for nx, L_, g_ in ((1e4, 117.84375, 6.2e-6), (10 ** 4.5, 130.75, 2.5e-5)):   # outside the gate set; checked
        pts.setdefault((round(np.log10(nx), 6), round(L_, 6)), dict(n_X=nx, L=L_, rel=-g_, separate=True))  # separately (STATUS 12)
    key = {(round(float(np.log10(x)), 6), round(float(y), 6)): i for i, (x, y) in enumerate(zip(d["n_X"], d["L"]))}
    from qml.certify import groups
    G, G0 = groups(d), groups(d0)
    pos, pos0 = {int(i): k for k, i in enumerate(P["ev_idx"])}, {int(i): k for k, i in enumerate(P0["ev_idx"])}
    pqc = [k for k in P if k.startswith("dense__PQC-full__") and k.endswith("__k0")]
    rows = ["| n_X | L km | split | label source | label gain over cold | groups (repaired labels) | groups (original labels) | "
            "dense PQC-full regret, mean over seeds: repaired / original labels | cold-baseline regret (repaired labels) |",
            "|---|---|---|---|---|---|---|---|---|"]
    touched = {}
    for kk in sorted(pts, key=lambda t: (t[0], t[1])):
        i = key[kk]
        gs = [g for g, m in G.items() if m[i]] or ["not scored / not evaluated"]
        gs0 = [g for g, m in G0.items() if m[i]] or ["not scored / not evaluated"]
        for g in gs:
            if g in G:
                touched[g] = touched.get(g, 0) + 1
        r1 = np.mean([P[k][pos[i]] for k in pqc]) if i in pos else np.nan
        r0 = np.mean([P0[k][pos0[i]] for k in pqc]) if i in pos0 else np.nan
        rc = P["cold"][pos[i]] if i in pos else np.nan
        g_ = pts[kk]["rel"]
        rows.append(f"| {d['n_X'][i]:.3g} | {d['L'][i]:.4f} | {d['split'][i]} | {d['source'][i]} | "
                    f"{fmt(-g_)}{' (separate check, STATUS 12)' if pts[kk].get('separate') else ''} | {', '.join(gs)} | {', '.join(gs0)} | "
                    f"{fmt(r1)} / {fmt(r0)} | {fmt(rc)} |")
    return "\n".join(rows), touched, len(pts)


def d0_crossovers(lab):
    return lab["crossovers"]


def main():
    global SHOWN_MODELS, SHOWN_TRAINED
    rep_path = repaired_labels()
    d = dict(np.load(rep_path, allow_pickle=False))
    lab = json.load(open(rep_path.replace(".npz", ".json")))
    d0 = dict(np.load(sorted(glob.glob(os.path.join(C.DATA, "labels_*.npz")))[-1], allow_pickle=False))
    repair = json.load(open(os.path.join(ROOT, lab["repair"], "repair_summary.json")))
    sel = json.load(open(latest("qml_select_*/selection.json")))
    tune = json.load(open(latest("qml_tune_*/tune.json")))
    fin = json.load(open(latest("qml_final_*/final.json")))
    lat = json.load(open(latest("qml_latency_*/latency.json")))
    mf, mf0 = certifications()
    M, M0 = json.load(open(mf)), json.load(open(mf0))
    P = dict(np.load(os.path.join(os.path.dirname(mf), "perpoint.npz")))
    P0 = dict(np.load(os.path.join(os.path.dirname(mf0), "perpoint.npz")))
    od = os.path.dirname(mf)
    same_nn = nn_identical(fin)
    if same_nn:
        SHOWN_MODELS = [m for m in MODELS if m != "NN-matched"]
        SHOWN_TRAINED = [m for m in TRAINED if m != "NN-matched"]
    s12_table, s12_touched, s12_n = status12_points(d, d0, P, P0)
    figs = figures(M, P, d, od)
    useful, U = usefulness(M, lat)
    useful_alt, U_alt = usefulness(M, lat, T=M0["timing"])
    flips = [(a_, b_) for a_, b_ in zip(useful.splitlines()[2:], useful_alt.splitlines()[2:])
             if [c.strip().split(" (")[0].split(":")[0] for c in a_.split("|")[6:9]] !=
                [c.strip().split(" (")[0].split(":")[0] for c in b_.split("|")[6:9]]]
    T0, T1 = M0["timing"], M["timing"]
    timing_cmp = ["| batch-1 end-to-end median, µs | this certification (repaired labels) | original-label certification |",
                  "|---|---|---|"]
    for k_ in range(3):
        timing_cmp.append(f"| k = {k_ + 1} Newton steps | {T1['batch1_kstep'][k_]['end_to_end_us_median']:.0f} | "
                          f"{T0['batch1_kstep'][k_]['end_to_end_us_median']:.0f} |")
    timing_cmp += [f"| cold solve | {T1['batch1_cold']['end_to_end_us_median']:.0f} | {T0['batch1_cold']['end_to_end_us_median']:.0f} |",
                   f"| warm solve | {T1['batch1_warm']['end_to_end_us_median']:.0f} | {T0['batch1_warm']['end_to_end_us_median']:.0f} |"]
    from qml.certify import groups as _groups
    Gd = _groups(d)
    nkm = P["warm_neighbour_km"]
    nb_rows = []
    for g_ in ("test_bulk", "test_tail"):
        m_ = Gd[g_][P["ev_idx"]] & np.isfinite(nkm)
        vals_ = np.round(nkm[m_], 4)
        u_, c_ = np.unique(vals_, return_counts=True)
        top = sorted(zip(c_, u_), reverse=True)[:4]
        nb_rows.append(f"{g_}: median {np.median(vals_):.4g} km; most common " + ", ".join(f"{v:g} km ({100 * c / vals_.size:.0f} %)" for c, v in top))

    primary = ["test_bulk", "test_tail"]
    ref = ["ref_bulk", "ref_tail", "supp_ref_incl_cliff_bulk", "supp_ref_incl_cliff_tail"]
    held = [g for g in next(iter(M["keys"].values()))["groups"] if g.startswith("heldout")]
    extr = [g for g in next(iter(M["keys"].values()))["groups"] if g.startswith("extrap")]
    kill_vals = collect(M, "dense", "PQC-full", "test_bulk", "k0", "median")
    kill = float(np.mean(kill_vals)) > 1e-2
    kill_vals0 = collect(M0, "dense", "PQC-full", "test_bulk", "k0", "median")
    n_tb, n_tb0 = collect(M, "dense", "PQC-full", "test_bulk", "k0", "n"), collect(M0, "dense", "PQC-full", "test_bulk", "k0", "n")
    xo = lab["crossovers_repaired"]
    n_cliff = int(((d["split"] == "ref") & d["ref_cliff"]).sum())

    sel_rows = ["| family | config index | config | params | val median regret | val ≤1e-6 % | selected |", "|---|---|---|---|---|---|---|"]
    runs = {r["key"].replace("|", "__"): r for r in tune["runs"]}
    for key, r in sorted(sel["val"].items()):
        sel_rows.append(f"| {r['family']} | {r['config_index']} | {runs[key]['config']} | {r['n_params']} | {fmt(r['median'])} | "
                        f"{fmt(r['pct_le_1e-06'], False)} | {'**yes**' if sel['selected'][r['family']] == r['config_index'] else ''} |")
    lat_rows = ["| model | params | train s, CPU (seed 0, Apple M2 Pro) | train s, GPU (A100; mean ± sd, seeds 1-4) | "
                "batch-1 device µs | batch-1 end-to-end µs | batch-1024 throughput /s | backend |",
                "|---|---|---|---|---|---|---|---|"]
    for model in SHOWN_TRAINED + ["lookup-2D"]:
        l_ = lat["models"].get(model, {})
        rs = [r for r in fin["variants"]["dense"]["runs"] if r["model"] == model]
        cpu = [r["train_s"] for r in rs if r.get("device", {}).get("backend") == "cpu" and r.get("train_s") is not None]
        gpu = [r["train_s"] for r in rs if r.get("device", {}).get("backend") == "gpu" and r.get("train_s") is not None]
        lat_rows.append(f"| {model} | {rs[0]['n_params'] if rs else fin['variants']['dense']['lookup_nodes']['lookup_2D']} | "
                        f"{ms(cpu, False) if cpu else ('not recorded' if any(r.get('device', {}).get('backend') == 'cpu' for r in rs) else '–')} | {ms(gpu, False) if gpu else '–'} | "
                        f"{fmt(l_.get('batch1_device_us'), False)} | {fmt(l_.get('batch1_end_to_end_us'), False)} | "
                        f"{fmt(l_.get('throughput_per_s'))} | {l_.get('backend', '')} |")
    T = M["timing"]
    tim = (f"- Batch-1 Newton steps from a surrogate prediction, end-to-end median: k=1 {T['batch1_kstep'][0]['end_to_end_us_median']:.0f} µs, "
           f"k=2 {T['batch1_kstep'][1]['end_to_end_us_median']:.0f} µs, k=3 {T['batch1_kstep'][2]['end_to_end_us_median']:.0f} µs.\n"
           f"- Batch-1 cold solve (optimize, 2^12 Sobol, top-8): {T['batch1_cold']['end_to_end_us_median']:.0f} µs median "
           f"(p90 {T['batch1_cold']['end_to_end_us_p90']:.0f} µs).\n"
           f"- Batch-1 warm solve from the nearest lower training node: {T['batch1_warm']['end_to_end_us_median']:.0f} µs median, "
           f"{T['batch1_warm']['iters_median']:.0f} iterations.\n"
           f"- PQC on hardware: shots per prediction for a standard error ε in every ⟨Z_i⟩ (median over the query set): "
           + ", ".join(f"{k} → {v:.3g}" for k, v in lat["pqc_shots_estimate"].items()) + ". One circuit per prediction.")
    nbk = M["baselines"]["warm_neighbour_km"]
    A_ = os.path.join(C.QML, "slides", "assets")
    fig_list = [
        (os.path.join(od, "regret_cdf_dense.png"), "Regret CDF of the surrogate predictions (no Newton steps), test bulk and tail, dense training, all seeds",
         f"`{os.path.relpath(mf, ROOT)}` (repaired labels)"),
        (os.path.join(od, "regret_cdf_sparse.png"), "The same for sparse training (Amendment 3)", f"`{os.path.relpath(mf, ROOT)}`"),
        (os.path.join(A_, "showcase_predictions.png"), "Predicted vs optimal settings across distance, and the % of the optimal key "
         "rate each prediction delivers; n_X = 1e6 test points and the held-out n_X = 5e8; PQC-full, NN-tuned, lookup-2D, seed 0",
         "`qml/slides/make_showcase.py`, repaired labels and the primary certification"),
        (os.path.join(A_, "params_across_distance.png"), "Optimal settings across distance for n_X = 1e4, 1e6, 1e9 (labels only)",
         "`qml/slides/make_showcase.py`, repaired labels"),
        (os.path.join(A_, "params_by_blocksize.png"), "All five optimal settings across distance, one panel per block size, with "
         "every isolated jump marked", "`qml/slides/make_params_grid.py --labels <repaired>`"),
        (os.path.join(A_, "pqc_circuit_8q_labelled.png"), "One layer of the selected PQC-full circuit (8 qubits; repeated 6 times; "
         "5 answer qubits read out)", "`qml/slides/draw_circuit.py` (.venv-qml, PennyLane)"),
    ]
    paper_figs = "\n".join(f"- `{os.path.relpath(f, ROOT)}`: {what}. Source: {src}." for f, what, src in fig_list if os.path.exists(f))
    missing = [os.path.relpath(f, ROOT) for f, _, _ in fig_list if not os.path.exists(f)]
    if missing:
        paper_figs += "\n- Not present (not drawn): " + ", ".join(f"`{m}`" for m in missing) + "."
    sp = fin["variants"]["dense"]
    ridge_spec = [r for r in tune["runs"] if r["family"] == "RFF-ridge" and r["config_index"] == sel["selected"]["RFF-ridge"]][0]

    txt = f"""# QML surrogate study: results (Phase 4)

Branch `qml-surrogate`. Protocol: Section 7 of `result_report/final_report.tex` and `qml/PREREGISTRATION.md`
(Amendments 1–5). Every number below is read from the result files listed at the end. Every regret is computed
from R re-evaluated by the float64 `lim` model at the predicted parameters. No model-predicted R is used anywhere.

**No quantum advantage is claimed.** The circuits are simulated classically, and the comparison is against classical
models of matched size.

## Summary

- **Labels.** All tables use the **repaired labels** (Amendment 4; `{os.path.basename(rep_path)}`), unless a table
  says otherwise. The certification against the original labels is reported once, as a secondary table.
- **Kill criterion** (dense PQC-full median bulk test regret above 1e-2, mean over seeds): **{'MET: negative result' if kill else 'not met'}**.
  The value is {ms(kill_vals)} (mean ± sd over {len(kill_vals)} seeds; repaired labels). Against the original labels
  it is {ms(kill_vals0)}.
- The usefulness verdicts per model and scenario are in the usefulness section below. Its definitions are post hoc
  and stated there, with the raw numbers next to them.
{"- **NN-tuned and NN-matched are the same configuration in this run** (both h = 12), so they are one result. Only NN-tuned is shown." if same_nn else ""}
- **Cold-solve limit (STATUS 12).** At {s12_n} label points a warm sweep beats the 2^14-Sobol/top-16 cold solve, by up to
  3.7e-4. The labels there are the best known optima, not proven global optima. Where these points enter a reported
  number is listed in the section "Points where the cold solve is beaten".

## Setup actually run

| item | value |
|---|---|
| labels | `{lab['config']['nX'].__len__()}` block sizes, {lab['n_points']} points; quality gate against the reference: {lab['gate_pass']} |
| label repair (Amendment 4) | {repair['n_repaired']} labels replaced by the backward sweep; gate {repair['gate']['n_checked']} cold solves, 0 repaired labels worse; gate_pass {repair['gate_pass']}; crossovers {len(xo)} (original {len(d0_crossovers(lab))}) |
| training points | dense {fin['variants']['dense']['n_train']}, sparse {fin['variants']['sparse']['n_train'] if 'sparse' in fin['variants'] else '–'} |
| tuning | E = {tune['config']['E']} full-batch Adam steps, seed 0, 6 configurations per family |
| final runs | seeds {fin['config']['seeds']}, variants {fin['config']['variants']} |
| evaluation points | {M['config']['n_eval']} |

## Hyperparameter selection (validation split; median regret, bulk and tail pooled, crossover windows excluded)

{chr(10).join(sel_rows)}

The NN-matched model uses the NN configuration with the same index as the selected PQC-full configuration.

## Primary metric: test splits, dense training (mean ± sd over seeds; lookups are deterministic)

{table(M, 'dense', primary)}

### Reference points (the report's grid), and the supplementary score including the reference cliff_edge points

The `supp_ref_incl_cliff_*` rows are supplementary and were requested by the owner after the cliff_edge diagnosis
(STATUS 10a). The registered metric is the `ref_*` rows, which exclude those points. The reference flags 7 cliff_edge
points, but they are {n_cliff} distinct (n_X, L) pairs: two points of the main grid (`reference_lim.json`) are also
tail-grid points. All {n_cliff} lie in the tail region, so the bulk supplementary rows equal the `ref_bulk` rows by
construction.

{table(M, 'dense', ref)}

### Held-out block sizes (interpolation in n_X)

{table(M, 'dense', held)}

### Extrapolation (reported separately, never pooled)

{table(M, 'dense', extr)}

### Crossover windows (±1 km around the {len(xo)} crossovers of the repaired labels; Amendments 2 and 4) and the drift window (listed)

{table(M, 'dense', ['crossover_windows', 'drift_window_listed'])}

### Horizon error (km, predicted minus label; largest lattice L with R(p̂) > 0)

{horizon_table(M, 'dense')}

### Secondary: the same test splits certified against the original labels

`{os.path.relpath(mf0, ROOT)}`. The scored point sets differ: the repaired labels have {len(xo)} crossover windows
instead of {len(d0_crossovers(lab))}, so more points fall inside a window and leave the test splits (dense PQC-full test_bulk n:
{int(np.median(n_tb))} repaired vs {int(np.median(n_tb0))} original).

{table(M0, 'dense', primary)}

## Points where the cold solve is beaten (STATUS 12)

A cold solve (2^14 Sobol, top-16, seed 11) can miss the better of two optima next to a crossover. At these {s12_n} points
a warm sweep label beats it (two-sided check at the {repair['gate']['n_checked']} gate points, plus two points checked
separately). The label is used as the reference there; it is the best known optimum, not a proven global one. A
point enters a reported number only through the groups listed; groups are recomputed for each label file.
Groups touched (repaired labels): {', '.join(f'{g} ({n})' for g, n in sorted(s12_touched.items())) or 'none'}.

{s12_table}

## Hybrid: surrogate + k Newton steps (dense; test splits)

{chr(10).join(table(M, 'dense', primary, k).replace('| model |', f'| model (k={k[1]}) |', 1) for k in ('k1', 'k2', 'k3'))}

### Baselines for the two hybrid scenarios

{baseline_rows(M, primary + ['ref_bulk', 'ref_tail'] + held + extr)}

**Scenario (ii) neighbour distance (decision (a) at STOP 2).**
- The preregistration's text says the neighbour is "0.5 km away in the bulk and 0.125 km in the tail".
- The registered rule is the nearest lower training node. For test points that node is 0.25 or 0.375 km away in the
  bulk, and 0.0625 or 0.094 km in the tail. The reason is the warm-sweep node spacing: training nodes every 0.5 km
  in the bulk and every 0.125 km in the tail, with test points offset between them.
- Held-out and extrapolated block sizes have no training nodes. There the neighbour is the previous lattice label
  (0.125 km below 100 km, 1/32 km above).
- Measured on the test splits (repaired-label groups): {nb_rows[0]}; {nb_rows[1]}.
- Over all evaluation points pooled by region (test, ref, held-out and extrapolated together, so this mixes the
  rules): median bulk {nbk['bulk']['median']} km, tail {nbk['tail']['median']} km.

## Latency and cost (latency: dense seed-0 weights; training time: all seeds, per hardware)

{chr(10).join(lat_rows)}

{tim}

Batch-1 timings of the two certification runs (same code path, same machine, measured in each run):

{chr(10).join(timing_cmp)}

They differ by up to about 2×, and k = 3 in this run is out of line with k = 1, 2. The usefulness verdicts below use
this run's timings. Recomputed with the original-label run's timings, {('these rows change: ' + '; '.join(a_.split('|')[1].strip() + ' ' + a_.split('|')[2].strip() for a_, _ in flips)) if flips else 'no (i)/(ii) verdict changes'}.
- Batched baselines over all {M['config']['n_eval']} evaluation points (wall clock, this laptop): cold
  {M['baselines']['cold_batched_s']:.0f} s and warm {M['baselines']['warm_batched_s']:.0f} s in this certification;
  {M0['baselines']['cold_batched_s']:.0f} s and {M0['baselines']['warm_batched_s']:.0f} s in the original-label certification.
  The cold solve is the same computation in both, so wall-clock timings of this size vary by tens of percent between runs.
- Training time: CPU (seed 0, the stopped run on the Mac) and GPU (seeds 1-4, Leonardo A100) are separate columns and are
  never pooled.

## Usefulness under Section 7 (dense; test splits)

The definitions are **post hoc**: the preregistration does not define "regret class" or "comparable latency".
- **(b)** The model has a lower median regret than lookup-2D, a higher % within 1e-6, and a batch-1 end-to-end
  latency of at most 10× lookup-2D's ({U['lookup_us']:.0f} µs).
- **(i) and (ii)** For some k ≤ 3, the hybrid's % within 1e-6 is at least the cold solve's (i) or the warm solve's
  (ii). Its latency (surrogate batch-1 end-to-end + k-step batch-1 end-to-end) is also below the cold solve's
  ({U['cold_us']:.0f} µs) or the warm solve's ({U['warm_us']:.0f} µs) batch-1 end-to-end.

The two latencies are added across environments. Transfer between `.venv-qml` and `.venv` is not included.

{useful}

## Sparse training (Amendment 3; 2 km labels; its own tables, never pooled)

{table(M, 'sparse', primary + ['ref_bulk', 'ref_tail'] + held)}

## Fourier spectrum of the trained PQC-full (selected configuration, seed 0, from tuning)

- Encoding gates per input: {ridge_spec['spectrum']['n_encoding_gates_per_input']}.
- Largest reachable frequency: u_L {ridge_spec['spectrum']['max_frequency_uL']:.2f}, v {ridge_spec['spectrum']['max_frequency_v']:.2f}
  (sum of |trained input scalings|).
- The RFF ridge samples this spectrum with D/2 = {ridge_spec['D2']} frequencies. Its result is in the tables as
  `RFF-ridge`.

## Figures for the paper

These figures were built earlier for slides; the slide decks are no longer produced (owner decision, 2026-11-09), and
the figures are used as paper figures instead. Each one is listed with what it was drawn from.

{paper_figs}

## Limitations

- Circuits are simulated analytically (exact expectation values). Hardware shot noise and gate errors are not
  included; see the shot estimate above and Phase 5a.
- One block-size range [1e4, 1e9] and two inputs. Extended inputs are Phase 5b.
- The latency comparison crosses two environments and one CPU machine (`.venv-qml` jax 0.10.2; `.venv` jax 0.9.0).
- **cliff_edge.** For label points the registered rule flags 0 points. The reference's 7 flags ({n_cliff} distinct
  points) are converged optima (STATUS 10a); the supplementary rows show the effect of including them.
- **Selection.** One validation split and one seed per configuration; 2000 steps for every family. Other budgets may
  rank the families differently.

## Sources

- selection `{os.path.relpath(latest('qml_select_*/selection.json'), ROOT)}`
- tuning `{os.path.relpath(latest('qml_tune_*/tune.json'), ROOT)}`
- final `{os.path.relpath(latest('qml_final_*/final.json'), ROOT)}`
- latency `{os.path.relpath(latest('qml_latency_*/latency.json'), ROOT)}`
- metrics, primary (repaired labels) `{os.path.relpath(mf, ROOT)}`
- metrics, secondary (original labels) `{os.path.relpath(mf0, ROOT)}`
- labels `{os.path.relpath(rep_path, ROOT)}`; repair `{lab['repair']}`
- STATUS 12 checks `{os.path.relpath(latest('qml_coldcheck_*/two_sided.json'), ROOT)}`, `{os.path.relpath(latest('qml_suboptcheck_*/check.json'), ROOT)}`
- figures: {', '.join('`' + os.path.relpath(f, ROOT) + '`' for f in figs)}

No quantum advantage is claimed.
"""
    with open(os.path.join(C.QML, "RESULTS.md"), "w") as fh:
        fh.write(txt)
    print("wrote qml/RESULTS.md;", "kill criterion met" if kill else "kill criterion not met", flush=True)


if __name__ == "__main__":
    main()
