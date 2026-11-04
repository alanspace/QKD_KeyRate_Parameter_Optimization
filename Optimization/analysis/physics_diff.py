"""Item A/B: quantify every difference between src/qkd (legacy) and Lim et al. 2014, and check the
paper's regression targets. All numbers in PHYSICS_DIFF.md come from this script.

Part 1 -- cumulative chain legacy -> lim_exact, one switch at a time, each stage re-optimised
(cold, 2^14 Sobol, top-16) at L in {0, 50, 100, 150} km x n_X in {1e4, 1e9}. Stage 0 uses the
physics_lim implementation with legacy switches (verified equal to src/qkd to 1e-8 in tests).

Part 2 -- paper targets (Lim et al. 2014 text / Fig. 1, 15 % tolerance, never tuned):
  T1  R(L=0, n_X=1e9) ~ 1e-2
  T2  R(100 km, 1e9) / R(100 km, 1e7) ~ 1.75
  T3  n_X = 1e4 "reaches ~1e-7 at ~135 km": the L where the optimised R falls to 1e-7, ~135 km
      (sweep on a 1 km grid, then bisection on log R = -7 with warm-started re-optimisation)
for: lim_exact, the diagnostic variant lim_exact[emis_eta=sys], and legacy.

Usage: python Optimization/analysis/physics_diff.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from src.qkd_opt import LEGACY, LIM_EXACT, Preset, check_x64  # noqa: E402
from src.qkd_opt.optimize import optimize_batch, optimize_sweep, warm_solve, eval_starts  # noqa: E402
from src.qkd_opt.provenance import stamp, to_jsonable  # noqa: E402
import jax.numpy as jnp  # noqa: E402

TOL = 0.15


def chain():
    f0 = LEGACY.flags._replace(model="lim")
    c0 = LEGACY.consts
    steps = [("0 legacy (src/qkd switches)", {}, {}),
             ("1 m_Z = total Z errors (Eq. 4)", dict(mz_mode="errors"), {}),
             ("2 e_mis 0.01 -> 5e-3", {}, dict(e_mis=5e-3)),
             ("3 p_ap 0 -> 0.04 in e_k", {}, dict(p_ap=0.04)),
             ("4 R_k = D_k(1+p_ap) in counts/N", dict(afterpulse_in_rate=True), {}),
             ("5 eps_sec = kappa*l (kappa=1e-15)", dict(eps_mode="kappa"), {}),
             ("6 no clamp of n^-, m^- at 0", dict(clamp_minus=False), {}),
             ("7 eta_ch (paper text) in e_mis term = lim_exact", dict(emis_eta="ch"), {})]
    out, f, c = [], f0, c0
    for name, df, dc in steps:
        f, c = f._replace(**df), c._replace(**dc)
        out.append(Preset(name=name, flags=f, consts=c))
    assert out[-1].flags == LIM_EXACT.flags and out[-1].consts == LIM_EXACT.consts, (out[-1], LIM_EXACT)
    return out


def targets(pre):
    r = optimize_batch([0.0, 100.0, 100.0], [1e9, 1e9, 1e7], pre, n_sobol=16384, top_k=16)
    T1 = r["R"][0]
    T2 = r["R"][1] / r["R"][2] if r["R"][2] > 0 else None
    Lg = np.arange(60.0, 181.0, 1.0)
    sw = optimize_sweep(Lg, [1e4], pre, global_every=5, refine_horizon=True)
    R = sw["R"][0]
    above = np.where(R >= 1e-7)[0]
    T3 = None
    if len(above) and above[-1] < len(Lg) - 1:
        k = above[-1]
        lo, hi, th = Lg[k], Lg[k + 1], sw["theta"][0, k]
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            w = warm_solve(th[None], [mid], [1e4], pre)
            Rm = float(np.asarray(eval_starts(jnp.asarray(w["theta"])[:, None, :], jnp.asarray([mid]),
                                              jnp.asarray([1e4]), pre.consts, pre.flags)[0])[0, 0])
            if Rm >= 1e-7:
                lo, th = mid, w["theta"][0]
            else:
                hi = mid
        T3 = lo
    elif len(above) == 0:
        T3 = "R < 1e-7 on the whole 60-180 km grid"
    h = sw["horizon"][0].get("horizon_km")

    def chk(v, tgt):
        if not isinstance(v, (float, np.floating)):
            return "MISMATCH"
        return "ok" if abs(v / tgt - 1) <= TOL else "MISMATCH"
    return dict(T1_R0_1e9=T1, T1_status=chk(T1, 1e-2), T2_ratio=T2, T2_status=chk(T2, 1.75),
                T3_L_at_R_1e-7_nX1e4=T3, T3_status=chk(T3, 135.0), horizon_nX1e4_km=h)


def main():
    check_x64()
    cfg = dict(script="physics_diff.py", chain_points=dict(L=[0, 50, 100, 150], nX=[1e4, 1e9]), tol=TOL)
    st = stamp(cfg)
    L = np.array([0.0, 50.0, 100.0, 150.0] * 2)
    N = np.repeat([1e4, 1e9], 4)
    rows = []
    for pre in chain():
        r = optimize_batch(L, N, pre, n_sobol=16384, top_k=16)
        rows.append(dict(stage=pre.name, R=r["R"], params=r["params"], status=list(r["status"])))
        print(f"[diff] {pre.name}: {np.array2string(r['R'], precision=3)}", flush=True)
    diag = Preset("lim_exact[emis_eta=sys] (diagnostic, NOT a preset)", LIM_EXACT.flags._replace(emis_eta="sys"),
                  LIM_EXACT.consts)
    tg = {}
    for pre in (LIM_EXACT, diag, LEGACY):
        tg[pre.name] = targets(pre)
        print(f"[targets] {pre.name}: {tg[pre.name]}", flush=True)
    out_dir = os.path.join(ROOT, "results", f"physics_diff_{st['git_commit']}_{st['config_hash']}")
    os.makedirs(out_dir, exist_ok=True)
    res = dict(stamp=st, config=cfg, chain_L=L, chain_nX=N, chain=rows, targets=tg)
    with open(os.path.join(out_dir, "physics_diff.json"), "w") as fh:
        json.dump(to_jsonable(res), fh, indent=1)
    md = [f"# Physics differences and paper targets (generated)", "",
          f"`Optimization/analysis/physics_diff.py`, git `{st['git_commit']}`, config `{st['config_hash']}`, "
          f"physics hash `{st['physics_source_hash']}`.", "",
          "## Cumulative chain legacy → lim_exact (optimised R, bits/pulse)", "",
          "| stage | " + " | ".join(f"L={l:g}, n_X={n:.0e}" for l, n in zip(L, N)) + " |",
          "|---|" + "---|" * len(L)]
    for r in rows:
        md.append(f"| {r['stage']} | " + " | ".join(f"{x:.3e}" if x > 0 else f"({x:.1e}, l≤0)" for x in r["R"]) + " |")
    md += ["", "## Paper targets (15 % tolerance, never tuned)", "",
           "| model | T1 R(0,1e9) [~1e-2] | T2 ratio [~1.75] | T3 L at R=1e-7, n_X=1e4 [~135 km] | horizon n_X=1e4 (km) |",
           "|---|---|---|---|---|"]
    for k, v in tg.items():
        t3 = v["T3_L_at_R_1e-7_nX1e4"]
        t3s = f"{t3:.2f}" if isinstance(t3, float) else str(t3)
        md.append(f"| {k} | {v['T1_R0_1e9']:.4e} ({v['T1_status']}) | "
                  f"{'–' if v['T2_ratio'] is None else f'{v['T2_ratio']:.3f}'} ({v['T2_status']}) | "
                  f"{t3s} ({v['T3_status']}) | {v['horizon_nX1e4_km']} |")
    with open(os.path.join(out_dir, "PHYSICS_DIFF_TABLES.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
