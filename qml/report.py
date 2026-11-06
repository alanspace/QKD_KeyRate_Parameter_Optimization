"""qml/RESULTS.md and figures from the Phase 4 result files (main .venv). No number is typed by hand.

Inputs (latest of each): results/qml_select_*/selection.json, qml_tune_*/tune.json, qml_final_*/final.json,
qml_latency_*/latency.json, qml_certify_final_*/{metrics.json, perpoint.npz}, qml/data/labels_*.json.
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
K = ["k0", "k1", "k2", "k3"]


def latest(p):
    fs = sorted(glob.glob(os.path.join(C.RESULTS, p)), key=os.path.getmtime)
    if not fs:
        sys.exit(f"missing {p}")
    return fs[-1]


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
    for model in MODELS:
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
    for model in MODELS:
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
            for model, col in zip(MODELS + ["cold", "warm"], plt.cm.tab10.colors):
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


def usefulness(M, lat, var="dense"):
    """Post-hoc operationalisation of Section 7's criterion (stated in RESULTS.md); raw numbers are tabulated too."""
    T = M["timing"]
    kstep = {f"k{t['k']}": t["end_to_end_us_median"] for t in T["batch1_kstep"]}
    cold_us, warm_us = T["batch1_cold"]["end_to_end_us_median"], T["batch1_warm"]["end_to_end_us_median"]
    lk = lat["models"]["lookup-2D"]["batch1_end_to_end_us"]
    rows = []
    for model in TRAINED + ["lookup-2D"]:
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


def main():
    d = dict(np.load(sorted(glob.glob(os.path.join(C.DATA, "labels_*.npz")))[-1], allow_pickle=False))
    lab = json.load(open(sorted(glob.glob(os.path.join(C.DATA, "labels_*.json")))[-1]))
    sel = json.load(open(latest("qml_select_*/selection.json")))
    tune = json.load(open(latest("qml_tune_*/tune.json")))
    fin = json.load(open(latest("qml_final_*/final.json")))
    lat = json.load(open(latest("qml_latency_*/latency.json")))
    mf = latest("qml_certify_final_*/metrics.json")
    M = json.load(open(mf))
    P = dict(np.load(os.path.join(os.path.dirname(mf), "perpoint.npz")))
    od = os.path.dirname(mf)
    figs = figures(M, P, d, od)
    useful, U = usefulness(M, lat)

    primary = ["test_bulk", "test_tail"]
    ref = ["ref_bulk", "ref_tail", "supp_ref_incl_cliff_bulk", "supp_ref_incl_cliff_tail"]
    held = [g for g in next(iter(M["keys"].values()))["groups"] if g.startswith("heldout")]
    extr = [g for g in next(iter(M["keys"].values()))["groups"] if g.startswith("extrap")]
    kill_vals = collect(M, "dense", "PQC-full", "test_bulk", "k0", "median")
    kill = float(np.mean(kill_vals)) > 1e-2

    sel_rows = ["| family | config index | config | params | val median regret | val ≤1e-6 % | selected |", "|---|---|---|---|---|---|---|"]
    runs = {r["key"].replace("|", "__"): r for r in tune["runs"]}
    for key, r in sorted(sel["val"].items()):
        sel_rows.append(f"| {r['family']} | {r['config_index']} | {runs[key]['config']} | {r['n_params']} | {fmt(r['median'])} | "
                        f"{fmt(r['pct_le_1e-06'], False)} | {'**yes**' if sel['selected'][r['family']] == r['config_index'] else ''} |")
    lat_rows = ["| model | params | train s (mean) | batch-1 device µs | batch-1 end-to-end µs | batch-1024 throughput /s | backend |",
                "|---|---|---|---|---|---|---|"]
    for model in TRAINED + ["lookup-2D"]:
        l_ = lat["models"].get(model, {})
        rs = [r for r in fin["variants"]["dense"]["runs"] if r["model"] == model]
        lat_rows.append(f"| {model} | {rs[0]['n_params'] if rs else fin['variants']['dense']['lookup_nodes']['lookup_2D']} | "
                        f"{fmt(np.mean([r.get('train_s', np.nan) for r in rs]), False) if rs else '–'} | "
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
    sp = fin["variants"]["dense"]
    ridge_spec = [r for r in tune["runs"] if r["family"] == "RFF-ridge" and r["config_index"] == sel["selected"]["RFF-ridge"]][0]

    txt = f"""# QML surrogate study: results (Phase 4)

Branch `qml-surrogate`. Protocol: Section 7 of `result_report/final_report.tex` and `qml/PREREGISTRATION.md`
(Amendments 1–3). Every number below is read from the result files listed at the end. Every regret is computed
from R re-evaluated by the float64 `lim` model at the predicted parameters. No model-predicted R is used anywhere.

**No quantum advantage is claimed.** The circuits are simulated classically, and the comparison is against classical
models of matched size.

## Summary

- **Kill criterion** (PQC-full median bulk test regret above 1e-2, mean over seeds): **{'MET: negative result' if kill else 'not met'}**.
  The value is {fmt(float(np.mean(kill_vals)))}.
- The usefulness verdicts per model and scenario are in the usefulness section below. Its definitions are post hoc
  and stated there, with the raw numbers next to them.

## Setup actually run

| item | value |
|---|---|
| labels | `{lab['config']['nX'].__len__()}` block sizes, {lab['n_points']} points; quality gate against the reference: {lab['gate_pass']} |
| training points | dense {fin['variants']['dense']['n_train']}, sparse {fin['variants']['sparse']['n_train'] if 'sparse' in fin['variants'] else '–'} |
| tuning | E = {tune['config']['E']} full-batch Adam steps, seed 0, 6 configurations per family |
| final runs | seeds {fin['config']['seeds']}, variants {fin['config']['variants']} |
| evaluation points | {M['config']['n_eval']} |

## Hyperparameter selection (validation split; median regret, bulk and tail pooled, crossover windows excluded)

{chr(10).join(sel_rows)}

The NN-matched model uses the NN configuration with the same index as the selected PQC-full configuration.

## Primary metric: test splits, dense training (mean ± sd over seeds; lookups are deterministic)

{table(M, 'dense', primary)}

### Reference points (the report's grid), and the supplementary score including the 7 reference cliff_edge points

The `supp_ref_incl_cliff_*` rows are supplementary and were requested by the owner after the cliff_edge diagnosis
(STATUS 10a). The registered metric is the `ref_*` rows, which exclude those points.

{table(M, 'dense', ref)}

### Held-out block sizes (interpolation in n_X)

{table(M, 'dense', held)}

### Extrapolation (reported separately, never pooled)

{table(M, 'dense', extr)}

### Crossover windows (±1 km around the 13 below-horizon switches; Amendment 2) and the drift window (listed)

{table(M, 'dense', ['crossover_windows', 'drift_window_listed'])}

### Horizon error (km, predicted minus label; largest lattice L with R(p̂) > 0)

{horizon_table(M, 'dense')}

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
- Measured median distance: bulk {nbk['bulk']['median']} km, tail {nbk['tail']['median']} km. Distinct values
  (first 12): bulk {nbk['bulk']['values']}, tail {nbk['tail']['values']}.

## Latency and cost (dense, seed 0)

{chr(10).join(lat_rows)}

{tim}

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

## Limitations

- Circuits are simulated analytically (exact expectation values). Hardware shot noise and gate errors are not
  included; see the shot estimate above and Phase 5a.
- One block-size range [1e4, 1e9] and two inputs. Extended inputs are Phase 5b.
- The latency comparison crosses two environments and one CPU machine (`.venv-qml` jax 0.10.2; `.venv` jax 0.9.0).
- **cliff_edge.** For label points the registered rule flags 0 points. The reference's 7 flagged points are converged
  optima (STATUS 10a); the supplementary rows show the effect of including them.
- **Selection.** One validation split and one seed per configuration; 2000 steps for every family. Other budgets may
  rank the families differently.

## Sources

- selection `{os.path.relpath(latest('qml_select_*/selection.json'), ROOT)}`
- tuning `{os.path.relpath(latest('qml_tune_*/tune.json'), ROOT)}`
- final `{os.path.relpath(latest('qml_final_*/final.json'), ROOT)}`
- latency `{os.path.relpath(latest('qml_latency_*/latency.json'), ROOT)}`
- metrics `{os.path.relpath(mf, ROOT)}`
- figures: {', '.join('`' + os.path.relpath(f, ROOT) + '`' for f in figs)}

No quantum advantage is claimed.
"""
    with open(os.path.join(C.QML, "RESULTS.md"), "w") as fh:
        fh.write(txt)
    print("wrote qml/RESULTS.md;", "kill criterion met" if kill else "kill criterion not met", flush=True)


if __name__ == "__main__":
    main()
