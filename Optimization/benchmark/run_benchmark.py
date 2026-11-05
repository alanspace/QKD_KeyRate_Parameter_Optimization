"""Single, reproducible benchmark of optimisation methods against the trusted reference.

Methods
  reference            Optimization/reference/reference_<preset>.json (build_reference.py)
  new_cold             qkd_opt.optimize_batch at every reference point (2^12 Sobol, top-8, seed 7)
                       (seed 7, not 0/1: a scrambled Sobol sequence with the reference's seed would be a
                       prefix of the reference's own sample, i.e. not an independent comparison)
  new_warm             qkd_opt.optimize_sweep over the full legacy grid linspace(0,200,400), all n_X
  lookup_table         params linearly interpolated in L from a table of new_warm optima on every 4th
                       grid point offset by 2 (so every reference point lies between two table nodes)
  legacy_DA / legacy_Hessian / legacy_Hessian_FoR
                       frozen legacy outputs (legacy preset only), R re-evaluated from stored params

Metrics (points with R_ref > 0 only)
  regret = (R_ref - R_method) / R_ref: median, p99, max; % of points with regret <= 1e-6;
  count of points where a method beats the reference by > 1e-9 (each one is investigated, not celebrated)
  horizon per n_X: largest L with l > 0 (bisection for reference/new_warm; last grid point for legacy)
  smoothness, two windows per n_X (tau = 1e-3):
      S    = {L : R_ref(L) > tau * R_ref(0), L < reference horizon}
      tail = {L : L in [max S, reference horizon)}
      TV/km = sum |dp_norm|_1 / km over consecutive grid pairs inside the window, p_norm = p / box width,
              EXCLUDING pairs that are branch-switch events;
      branch-switch event = a pair (j, j+1) where new_warm changes branch (branch_id) within +/- 1
              grid point AND the method's R at j and j+1 is within 1e-6 relative of new_warm's
              (i.e. the method is on the optimal branch on both sides). Events are listed, not counted.
      jumps = remaining pairs with |dp_norm|_inf > 0.05 (listed with L) -- these are roughness.
  tail regret: against the dense tail reference (reference_<preset>_tail.json, every 0.5 km grid
      point from horizon - 15 km to horizon + 1 km), when present.
  timing: wall time and time per point; JIT compile reported separately (first call - second call
      in the same process, identical inputs); hardware stamped.

Usage: python Optimization/benchmark/run_benchmark.py --preset lim|lim_literal|legacy [--out DIR]
"""
import argparse
import glob
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from src.qkd_opt import get_preset, check_x64, reparam  # noqa: E402
from src.qkd_opt.objective import evaluate  # noqa: E402
from src.qkd_opt.optimize import optimize_batch, optimize_sweep  # noqa: E402
from src.qkd_opt.provenance import stamp, stamp_figure, to_jsonable  # noqa: E402
import legacy_runs  # noqa: E402

NX = legacy_runs.NX
GRID = legacy_runs.GRID
TAU = 1e-3
JUMP = 0.05
BOX = np.array([reparam.MU1_MAX, reparam.MU2_MAX, 1.0, 1.0, 1.0])
# categorical slots 1-6 of the validated palette (fixed order, one per method), plus ink for reference
COLORS = {"new_warm": "#2a78d6", "new_cold": "#eb6834", "lookup_table": "#1baf7a",
          "legacy_DA": "#eda100", "legacy_Hessian": "#e87ba4", "legacy_Hessian_FoR": "#008300"}
STYLE = {"new_warm": "-", "new_cold": "none", "lookup_table": ":", "legacy_DA": "--",
         "legacy_Hessian": "-.", "legacy_Hessian_FoR": (0, (5, 1, 1, 1))}
MARK = {"new_warm": None, "new_cold": "o", "lookup_table": "s", "legacy_DA": None, "legacy_Hessian": None,
        "legacy_Hessian_FoR": None}
INK = "#0b0b0b"


def _eval_RL(P, L, n, preset):
    f = jax.jit(jax.vmap(lambda p, l_, m: evaluate(p, l_, m, preset.consts, preset.flags)))
    o = f(jnp.asarray(P), jnp.asarray(L), jnp.asarray(n))
    return np.asarray(o["R"]), np.asarray(o["l"])


def regret_stats(R_ref, R_m, exclude=None):
    """Regret statistics over points with R_ref > 0, excluding `exclude` (cliff_edge reference points)."""
    pos = R_ref > 0
    if exclude is not None:
        pos = pos & ~np.asarray(exclude, bool)
    if not pos.any():   # e.g. every positive point of this subset is cliff_edge
        nan = float("nan")
        return dict(n=0, median=nan, p99=nan, max=nan, pct_within_1e6=nan, pct_within_1e4=nan, pct_within_1e3=nan,
                    pct_within_1e2=nan, n_beats_ref=0, min=nan)
    r = (R_ref[pos] - R_m[pos]) / R_ref[pos]
    r = np.where(np.isfinite(r), r, np.inf)
    return dict(n=int(pos.sum()), median=float(np.median(r)), p99=float(np.quantile(r, 0.99)),
                max=float(np.max(r)), pct_within_1e6=float(100 * np.mean(r <= 1e-6)),
                pct_within_1e4=float(100 * np.mean(r <= 1e-4)), pct_within_1e3=float(100 * np.mean(r <= 1e-3)),
                pct_within_1e2=float(100 * np.mean(r <= 1e-2)),
                n_beats_ref=int(np.sum(r < -1e-9)), min=float(np.min(r)))


def smoothness(params_row, L, mask, event_pairs):
    """TV/km and jumps on `mask`, with branch-switch event pairs excluded from both."""
    pn = params_row / BOX
    d = np.abs(np.diff(pn, axis=0))
    pair = mask[1:] & mask[:-1]
    ev = np.zeros_like(pair)
    ev[list(event_pairs)] = True
    ev &= pair
    keep = pair & ~ev
    if pair.sum() == 0:
        return dict(TV_per_km=None, n_jumps=0, jumps_at_km=[], branch_switch_events_at_km=[])
    span = float(np.sum(np.diff(L)[pair]))
    tv = float(np.sum(d[keep].sum(axis=1)) / span)
    jumps = np.where(keep & (d.max(axis=1) > JUMP))[0]
    return dict(TV_per_km=tv, n_jumps=int(len(jumps)), jumps_at_km=[round(float(L[j]), 3) for j in jumps],
                branch_switch_events_at_km=[round(float(L[j]), 3) for j in np.where(ev)[0]])


def event_pairs_for(P_method_row, R_method_row, sw_branch_row, R_new_row, is_new_warm=False, tol=1e-6):
    """Branch-switch event pairs (see module doc). new_warm: exactly the pairs where branch_id changes.
    Other methods: per new_warm switch, at most one pair within +/- 1 grid point where the method is
    within `tol` of new_warm on both sides -- the one with the largest parameter jump."""
    sw_pairs = np.where(np.diff(sw_branch_row) != 0)[0]
    if is_new_warm:
        return set(int(j) for j in sw_pairs)
    jump = np.abs(np.diff(P_method_row / BOX, axis=0)).max(axis=1)
    out = set()
    for j0 in sw_pairs:
        cand = [j for j in (j0 - 1, j0, j0 + 1) if 0 <= j < len(R_new_row) - 1 and
                all(R_new_row[k] > 0 and abs(R_method_row[k] - R_new_row[k]) <= tol * abs(R_new_row[k])
                    for k in (j, j + 1))]
        if cand:
            out.add(int(max(cand, key=lambda j: jump[j])))
    return out


def region_S(ref_L, ref_R, ref_horizon):
    """S on GRID from log-linear interpolation of the reference (positive part)."""
    pos = ref_R > 0
    if pos.sum() < 2:
        return np.zeros_like(GRID, bool)
    logR = np.interp(GRID, ref_L[pos], np.log10(ref_R[pos]), right=-np.inf)
    thr = np.log10(TAU * ref_R[0])
    S = logR > thr
    if ref_horizon is not None:
        S &= GRID < ref_horizon
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--global-every", type=int, default=10)
    a = ap.parse_args()
    preset = get_preset(a.preset)
    backend = check_x64()
    ref_path = a.reference or os.path.join(ROOT, "Optimization", "reference", f"reference_{preset.name}.json")
    ref = json.load(open(ref_path))
    cfg = dict(script="run_benchmark.py", preset=preset.name, reference=os.path.relpath(ref_path, ROOT),
               reference_config_hash=ref["stamp"]["config_hash"], tau=TAU, jump=JUMP,
               new_cold=dict(n_sobol=4096, top_k=8, seed=7),
               new_warm=dict(global_every=a.global_every, n_sobol_init=16384, top_k_init=16, n_sobol_check=2048,
                             top_k_check=4, seed=7),
               lookup_nodes="GRID[2::4]", flags=preset.flags._asdict(), consts=preset.consts._asdict())
    st = stamp(cfg)
    out_dir = a.out or os.path.join(ROOT, "results", f"benchmark_{preset.name}_{st['git_commit']}_{st['config_hash']}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[bench] {preset.name} -> {os.path.relpath(out_dir, ROOT)}", flush=True)

    pts = ref["points"]
    rL = np.array([p["L"] for p in pts])
    rN = np.array([p["n_X"] for p in pts])
    rR = np.array([p["R"] for p in pts], float)
    rP = np.array([p["params"] for p in pts], float)
    rC = np.array([bool(p.get("cliff_edge", False)) for p in pts])
    ref_idx = np.array([int(np.argmin(np.abs(GRID - x))) for x in rL])
    ref_h = {h["n_X"]: h.get("horizon_km") for h in ref["horizons"]}

    results, timing = {}, {}

    # ---- new_cold (timed twice: compile = first - second) ----
    t = time.perf_counter(); optimize_batch(rL, rN, preset, **cfg["new_cold"]); t1 = time.perf_counter() - t
    t = time.perf_counter(); cold = optimize_batch(rL, rN, preset, **cfg["new_cold"]); t2 = time.perf_counter() - t
    timing["new_cold"] = dict(first_call_s=t1, steady_s=t2, compile_overhead_s=t1 - t2,
                              n_points=len(rL), steady_per_point_ms=1e3 * t2 / len(rL),
                              note="batched: all reference points in one vectorised call")
    results["new_cold"] = dict(R=cold["R"], params=cold["params"], status=list(cold["status"]))
    print(f"[bench] new_cold first {t1:.1f}s steady {t2:.1f}s", flush=True)

    # ---- new_warm sweep (timed twice) ----
    t = time.perf_counter(); optimize_sweep(GRID, NX, preset, **cfg["new_warm"]); t1 = time.perf_counter() - t
    t = time.perf_counter(); sw = optimize_sweep(GRID, NX, preset, **cfg["new_warm"]); t2 = time.perf_counter() - t
    timing["new_warm"] = dict(first_call_s=t1, steady_s=t2, compile_overhead_s=t1 - t2,
                              n_points=int(GRID.size * len(NX)), steady_per_point_ms=1e3 * t2 / (GRID.size * len(NX)),
                              note="6 n_X chains advanced together; includes global checks, backfills, horizon bisection")
    nx_index = {nx: i for i, nx in enumerate(NX)}
    rows = np.array([nx_index[x] for x in rN])
    results["new_warm"] = dict(R=sw["R"][rows, ref_idx], params=sw["params"][rows, ref_idx],
                               status=[sw["status"][i, j] for i, j in zip(rows, ref_idx)])
    print(f"[bench] new_warm first {t1:.1f}s steady {t2:.1f}s; events: "
          f"{sum(e['type'] == 'branch_switch' for e in sw['events'])} switches", flush=True)

    # ---- lookup table from new_warm optima on GRID[2::4] ----
    nodes = np.arange(2, GRID.size, 4)
    LT = np.zeros((len(rL), 5))
    for k, (x, i) in enumerate(zip(rL, rows)):
        LT[k] = [np.interp(x, GRID[nodes], sw["params"][i, nodes, c]) for c in range(5)]
    R_lt, _ = _eval_RL(LT, rL, rN, preset)
    results["lookup_table"] = dict(R=R_lt, params=LT, status=["interpolated"] * len(rL))

    # ---- legacy runs (legacy preset only) ----
    legacy_full = {}
    if preset.name == "legacy":
        for name in ("legacy_DA", "legacy_Hessian", "legacy_Hessian_FoR"):
            run = legacy_runs.load(name)
            Rr, lr_ = legacy_runs.reevaluate(run, preset)
            legacy_full[name] = dict(R=Rr, l=lr_, params=run["params"], sha256=run["sha256"])
            results[name] = dict(R=Rr[rows, ref_idx], params=run["params"][rows, ref_idx],
                                 status=["stored"] * len(rL))

    # ---- metrics ----
    table = {}
    for m, r in results.items():
        table[m] = {"all": regret_stats(rR, np.asarray(r["R"]), rC)}
        for nx in NX:
            sel = rN == nx
            table[m][f"{nx:.0e}"] = regret_stats(rR[sel], np.asarray(r["R"])[sel], rC[sel])
    cliff_main = [dict(L=float(rL[k]), n_X=float(rN[k]), R_ref=float(rR[k]),
                       regret={m: float((rR[k] - np.asarray(r["R"])[k]) / rR[k]) for m, r in results.items()})
                  for k in np.where(rC)[0]]

    horizons = {}
    for i, nx in enumerate(NX):
        h = dict(reference=ref_h.get(nx))
        h["new_warm"] = sw["horizon"][i].get("horizon_km")
        for name, d in legacy_full.items():
            pos = np.where(d["l"][i] > 0)[0]
            h[name] = float(GRID[pos[-1]]) if len(pos) else None
        horizons[f"{nx:.0e}"] = h

    smooth = {}
    for i, nx in enumerate(NX):
        sel = rN == nx
        hz = ref_h.get(nx)
        S = region_S(rL[sel], rR[sel], hz)
        T = np.zeros_like(S)
        if S.any() and hz is not None:
            T = (GRID >= GRID[S].max()) & (GRID < hz)
        rec = dict(S_km=[float(GRID[S].min()), float(GRID[S].max())] if S.any() else None,
                   tail_km=[float(GRID[T].min()), float(GRID[T].max())] if T.any() else None)
        curves = {"new_warm": (sw["params"][i], sw["R"][i])}
        for name, d in legacy_full.items():
            curves[name] = (d["params"][i], d["R"][i])
        for m, (P, Rm) in curves.items():
            ev = event_pairs_for(P, Rm, sw["branch_id"][i], sw["R"][i], is_new_warm=(m == "new_warm"))
            rec[m] = dict(S=smoothness(P, GRID, S, ev), tail=smoothness(P, GRID, T, ev))
        smooth[f"{nx:.0e}"] = rec

    # ---- dense tail reference ----
    tail = None
    tpath = os.path.join(ROOT, "Optimization", "reference", f"reference_{preset.name}_tail.json")
    if os.path.exists(tpath):
        tref = json.load(open(tpath))
        tL = np.array([p["L"] for p in tref["points"]])
        tN = np.array([p["n_X"] for p in tref["points"]])
        tR = np.array([p["R"] for p in tref["points"]], float)
        tC = np.array([bool(p.get("cliff_edge", False)) for p in tref["points"]])
        tI = np.array([p["grid_index"] for p in tref["points"]])
        trows = np.array([nx_index[x] for x in tN])
        tc = optimize_batch(tL, tN, preset, **cfg["new_cold"])
        tres = {"new_cold": tc["R"], "new_warm": sw["R"][trows, tI]}
        LTt = np.zeros((len(tL), 5))
        for k, (x, i) in enumerate(zip(tL, trows)):
            LTt[k] = [np.interp(x, GRID[nodes], sw["params"][i, nodes, c]) for c in range(5)]
        tres["lookup_table"] = _eval_RL(LTt, tL, tN, preset)[0]
        for name, d in legacy_full.items():
            tres[name] = d["R"][trows, tI]
        tail = dict(reference=os.path.relpath(tpath, ROOT), reference_config_hash=tref["stamp"]["config_hash"],
                    reference_summary=tref["summary"], n_points=len(tL),
                    regret={m: {"all": regret_stats(tR, np.asarray(v), tC),
                                **{f"{nx:.0e}": regret_stats(tR[tN == nx], np.asarray(v)[tN == nx], tC[tN == nx])
                                   for nx in NX if np.any((tN == nx) & (tR > 0) & ~tC)}}
                            for m, v in tres.items()},
                    cliff_edge=[dict(L=float(tL[k]), n_X=float(tN[k]), R_ref=float(tR[k]),
                                     regret={m: float((tR[k] - np.asarray(v)[k]) / tR[k]) for m, v in tres.items()})
                                for k in np.where(tC)[0]],
                    tail_values=dict(L=tL, n_X=tN, R_ref=tR, cliff=tC, **{m: np.asarray(v) for m, v in tres.items()}))

    xcheck = {}
    for tag in ("", "_tail"):
        for dpath in sorted(glob.glob(os.path.join(ROOT, "results", f"crosscheck_{preset.name}{tag}_*", "crosscheck.json"))):
            xc = json.load(open(dpath))
            if xc["config"]["reference_config_hash"] in (ref["stamp"]["config_hash"],
                                                       (tail or {}).get("reference_config_hash")):
                xcheck["main" if not tag else "tail"] = dict(path=os.path.relpath(dpath, ROOT), **xc["summary"])

    summary = dict(stamp=st, config=cfg, backend=backend, regret=table, cliff_edge_main=cliff_main,
                   tail_regret=tail, crosscheck=xcheck,
                   main_values=dict(L=rL, n_X=rN, R_ref=rR, cliff=rC, **{m: np.asarray(r['R']) for m, r in results.items()}),
                   horizons=horizons, smoothness=smooth,
                   timing=timing, sweep_events=sw["events"],
                   new_warm_status_counts={s: int(np.sum(sw["status"] == s)) for s in np.unique(sw["status"])},
                   reference_summary=ref["summary"])
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(to_jsonable(summary), fh, indent=1)
    np.savez_compressed(os.path.join(out_dir, "new_warm_sweep.npz"), L=GRID, nX=np.array(NX), R=sw["R"], l=sw["l"],
                        params=sw["params"], branch_id=sw["branch_id"], status=sw["status"].astype(str),
                        grad_norm_theta=sw["grad_norm_theta"], hess_eig=sw["hess_eig"])
    write_markdown(out_dir, preset, summary)
    make_figures(out_dir, st, preset, rL, rN, rR, results, sw, legacy_full, smooth)
    print(f"[bench] done -> {os.path.relpath(out_dir, ROOT)}")


def fmt(x, p=2):
    if x is None:
        return "–"
    if isinstance(x, str):
        return x
    if not np.isfinite(x):
        return "inf"
    return f"{x:.{p}e}" if (abs(x) < 1e-2 and x != 0) or abs(x) >= 1e4 else f"{x:.{p}f}"


def write_markdown(out_dir, preset, s):
    st = s["stamp"]
    L = [f"# Benchmark — preset `{preset.name}`", "",
         f"Generated by `Optimization/benchmark/run_benchmark.py`. git `{st['git_commit']}`"
         f"{' (src/ has uncommitted changes)' if st['src_uncommitted_changes'] else ''}, "
         f"config hash `{st['config_hash']}`, physics source hash `{st['physics_source_hash']}`.",
         f"Hardware: {st['cpu']} ({st['n_cpu']} logical CPUs), {st['os']}; JAX {st['jax']} backend "
         f"`{st['jax_backend']}`, x64={st['x64']}.", "",
         "## Key-rate regret vs reference (points with R_ref > 0)", "",
         "regret = (R_ref − R)/R_ref. `≤1e-6` = % of points within 1e-6 relative. `beats ref` = points where the "
         "method is better than the reference by > 1e-9 relative (a reference weakness, listed for investigation).", "",
         "| method | n | median | p99 | max | ≤1e-6 (%) | ≤1e-4 (%) | ≤1e-3 (%) | ≤1e-2 (%) | beats ref |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for m, d in s["regret"].items():
        a = d["all"]
        L.append(f"| {m} | {a['n']} | {fmt(a['median'])} | {fmt(a['p99'])} | {fmt(a['max'])} | "
                 f"{a['pct_within_1e6']:.1f} | {a['pct_within_1e4']:.1f} | {a['pct_within_1e3']:.1f} | "
                 f"{a['pct_within_1e2']:.1f} | {a['n_beats_ref']} |")
    L += ["", f"Reference points flagged cliff_edge (R > 0, certificate gradient > 1e-3) are EXCLUDED from the "
          f"table above; their regret is listed here ({len(s['cliff_edge_main'])} points):"]
    for c in s["cliff_edge_main"]:
        L.append(f"- L={c['L']:.3f} km, n_X={c['n_X']:.0e}, R_ref={c['R_ref']:.4e}: " +
                 ", ".join(f"{m} {v:.2e}" for m, v in c["regret"].items()))
    L += ["", "Per n_X (median / max / ≤1e-6 %):", "", "| method | " + " | ".join(f"{nx:.0e}" for nx in NX) + " |",
          "|---|" + "---|" * len(NX)]
    for m, d in s["regret"].items():
        L.append(f"| {m} | " + " | ".join(
            f"{fmt(d[f'{nx:.0e}']['median'])} / {fmt(d[f'{nx:.0e}']['max'])} / {d[f'{nx:.0e}']['pct_within_1e6']:.0f}"
            for nx in NX) + " |")
    L += ["", "## Horizon (largest L with l > 0), km", "",
          "Reference and new_warm: bisection on the sign of the optimised l. Legacy: last point of the 0.5 km grid "
          "with re-evaluated l > 0 (resolution 0.5 km).", ""]
    meths = list(next(iter(s["horizons"].values())).keys())
    L += ["| n_X | " + " | ".join(meths) + " |", "|---|" + "---|" * len(meths)]
    for nx, h in s["horizons"].items():
        L.append(f"| {nx} | " + " | ".join("–" if h[m] is None else f"{h[m]:.3f}" for m in meths) + " |")
    if s.get("tail_regret"):
        t = s["tail_regret"]
        L += ["", f"## Tail regret vs the dense tail reference ({t['n_points']} grid points, horizon − 15 km … + 1 km)", "",
              f"Reference: `{t['reference']}` (config `{t['reference_config_hash']}`), summary {json.dumps(to_jsonable(t['reference_summary']))}.", "",
              "| method | n | median | p99 | max | ≤1e-6 (%) | ≤1e-4 (%) | ≤1e-3 (%) | ≤1e-2 (%) | beats ref |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for m, d in t["regret"].items():
            a = d["all"]
            L.append(f"| {m} | {a['n']} | {fmt(a['median'])} | {fmt(a['p99'])} | {fmt(a['max'])} | "
                     f"{a['pct_within_1e6']:.1f} | {a['pct_within_1e4']:.1f} | {a['pct_within_1e3']:.1f} | "
                     f"{a['pct_within_1e2']:.1f} | {a['n_beats_ref']} |")
        L += ["", f"Tail cliff_edge points (excluded above): {len(t['cliff_edge'])}"]
        for c in t["cliff_edge"]:
            L.append(f"- L={c['L']:.3f} km, n_X={c['n_X']:.0e}, R_ref={c['R_ref']:.4e}: " +
                     ", ".join(f"{m} {v:.2e}" for m, v in c["regret"].items()))
    if s.get("crosscheck"):
        L += ["", "## Independent solver cross-check of the reference (scipy SLSQP, random starts, p-space)", ""]
        for k, v in s["crosscheck"].items():
            L.append(f"- {k}: n={v['n']}, |d| ≤ 1e-6 at {v['pct_agree_1e6']:.1f} %, max d={fmt(v['max_d'])}, "
                     f"min d={fmt(v['min_d'])}, reference weaknesses={v['n_reference_weakness']} (`{v['path']}`)")
    L += ["", f"## Parameter smoothness (τ = {TAU:g})", "",
          "Windows: S = {L : R_ref(L) > τ·R_ref(0), L < horizon}; tail = [max S, horizon). TV/km of box-normalised "
          "parameters over consecutive 0.5 km grid pairs, branch-switch events excluded; jumps = remaining pairs with "
          f"max |Δp_norm| > {JUMP} (roughness). Branch-switch events (method switches together with new_warm and is "
          "optimal, within 1e-6, on both sides) are listed separately and are not roughness.", "",
          "| n_X | method | S: TV/km | S: jumps | tail: TV/km | tail: jumps (km) | branch-switch events (km) |",
          "|---|---|---|---|---|---|---|"]
    for nx, d in s["smoothness"].items():
        for m, v in d.items():
            if m in ("S_km", "tail_km"):
                continue
            evs = sorted(set(v["S"]["branch_switch_events_at_km"] + v["tail"]["branch_switch_events_at_km"]))
            L.append(f"| {nx} | {m} | {fmt(v['S']['TV_per_km'], 4)} | {v['S']['n_jumps']}"
                     f"{' at ' + str(v['S']['jumps_at_km'][:6]) if v['S']['n_jumps'] else ''} | "
                     f"{fmt(v['tail']['TV_per_km'], 4)} | {v['tail']['n_jumps']}"
                     f"{' at ' + str(v['tail']['jumps_at_km'][:8]) if v['tail']['n_jumps'] else ''} | {evs if evs else '–'} |")
        L.append(f"| {nx} | windows | S = {d['S_km']} | | tail = {d['tail_km']} | | |")
    L += ["", "## Timing (this machine, this run)", "",
          "| method | first call (s) | steady (s) | JIT/compile overhead (s) | points | steady ms/point | note |",
          "|---|---|---|---|---|---|---|"]
    for m, t in s["timing"].items():
        L.append(f"| {m} | {t['first_call_s']:.1f} | {t['steady_s']:.1f} | {t['compile_overhead_s']:.1f} | "
                 f"{t['n_points']} | {t['steady_per_point_ms']:.1f} | {t['note']} |")
    L += ["", "Legacy method timings are not re-run here; see `Optimization/benchmark/timing_benchmark.py`.", "",
          "## Continuation events (new_warm)", ""]
    for e in s["sweep_events"]:
        if e["type"] in ("branch_switch", "fallback", "backfill"):
            keep = {k: v for k, v in e.items() if k not in ("params_warm", "params_global")}
            L.append(f"- {json.dumps(to_jsonable(keep))}")
    L += ["", f"new_warm status counts: {s['new_warm_status_counts']}", "",
          f"Reference summary: {json.dumps(to_jsonable(s['reference_summary']))}"]
    with open(os.path.join(out_dir, "RESULTS.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")


def make_figures(out_dir, st, preset, rL, rN, rR, results, sw, legacy_full, smooth):
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "0.9", "grid.linewidth": 0.6, "font.size": 9})
    # Fig 1: key rate vs L per n_X
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    for i, (ax, nx) in enumerate(zip(axes.ravel(), NX)):
        R = sw["R"][i]
        ax.semilogy(GRID[R > 0], R[R > 0], color=COLORS["new_warm"], lw=2, label="new (warm sweep)")
        for name, d in legacy_full.items():
            Rr = d["R"][i]
            ax.semilogy(GRID[Rr > 0], Rr[Rr > 0], color=COLORS[name], lw=1.2, ls=STYLE[name], label=name)
        sel = (rN == nx) & (rR > 0)
        ax.semilogy(rL[sel], rR[sel], "o", ms=5, mfc="none", mec=INK, mew=1.2, label="reference")
        ax.set_title(f"n_X = 1e{int(np.log10(nx))}")
        if i % 3 == 0:
            ax.set_ylabel("secret key rate R (bits/pulse)")
        if i >= 3:
            ax.set_xlabel("fiber length L (km)")
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"Optimised key rate, preset {preset.name} (R > 0 shown)")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    stamp_figure(fig, st)
    fig.savefig(os.path.join(out_dir, "fig_keyrate.png"), dpi=200)
    plt.close(fig)

    # Fig 2: regret dot plot per method
    fig, ax = plt.subplots(figsize=(9, 4.2))
    methods = list(results)
    for k, m in enumerate(methods):
        R = np.asarray(results[m]["R"])
        pos = rR > 0
        r = (rR[pos] - R[pos]) / rR[pos]
        y = np.clip(np.where(np.isfinite(r), r, 1.0), 1e-16, 1.0)
        jitter = (np.random.default_rng(0).random(y.size) - 0.5) * 0.35
        ax.scatter(np.full(y.size, k) + jitter, y, s=10, color=COLORS[m], alpha=0.8, linewidths=0)
        ax.plot([k - 0.3, k + 0.3], [np.median(y)] * 2, color=INK, lw=1.5)
    ax.set_yscale("log")
    ax.axhline(1e-6, color="0.5", lw=0.8, ls="--")
    ax.text(len(methods) - 0.5, 1.3e-6, "1e-6", fontsize=7, color="0.4", ha="right")
    ax.set_xticks(range(len(methods)), methods, rotation=15)
    ax.set_ylabel("regret (R_ref − R)/R_ref, clipped to [1e-16, 1]")
    ax.set_title(f"Regret vs reference at {int((rR > 0).sum())} points (R_ref > 0), preset {preset.name}; bar = median")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    stamp_figure(fig, st)
    fig.savefig(os.path.join(out_dir, "fig_regret.png"), dpi=200)
    plt.close(fig)

    # Fig 3: parameter trajectories inside S for n_X = 1e4 and 1e9
    names = ["mu_1", "mu_2", "P_mu_1", "P_mu_2", "q_x (P_X)"]
    for nx in (1e4, 1e9):
        i = NX.index(nx)
        S_km = smooth[f"{nx:.0e}"]["S_km"]
        fig, axes = plt.subplots(5, 1, figsize=(8, 10), sharex=True)
        for c, ax in enumerate(axes):
            ax.plot(GRID, sw["params"][i, :, c], color=COLORS["new_warm"], lw=2, label="new (warm sweep)")
            for name, d in legacy_full.items():
                ax.plot(GRID, d["params"][i, :, c], color=COLORS[name], lw=1, ls=STYLE[name], label=name)
            if S_km:
                ax.axvspan(S_km[1], GRID[-1], color="0.93", zorder=0)
            ax.set_ylabel(names[c])
        axes[0].legend(fontsize=7, frameon=False, ncol=2)
        axes[0].set_title(f"Optimised parameters, n_X=1e{int(np.log10(nx))}, preset {preset.name} "
                          f"(grey: outside S, R < {TAU:g}·R(0) or beyond horizon)")
        axes[-1].set_xlabel("fiber length L (km)")
        fig.tight_layout(rect=(0, 0.02, 1, 1))
        stamp_figure(fig, st)
        fig.savefig(os.path.join(out_dir, f"fig_params_nx1e{int(np.log10(nx))}.png"), dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
