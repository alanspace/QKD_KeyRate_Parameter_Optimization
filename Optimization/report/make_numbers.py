"""Generate the numbers file imported by the report: result_report/generated_numbers.tex (LaTeX
macros) and results/NUMBERS.md (same values, with the source file of each). Reads
results/MANIFEST.json (written by Optimization/run_all.py) and the reference JSONs. Never edit the
outputs by hand -- re-run this script.

Usage: python Optimization/report/make_numbers.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

NXW = {1e4: "Four", 1e5: "Five", 1e6: "Six", 1e7: "Seven", 1e8: "Eight", 1e9: "Nine"}
PW = {"lim": "Lim", "lim_literal": "LimLiteral", "legacy": "Legacy"}


def sci(x, d=2):
    if x is None:
        return "--"
    if isinstance(x, str):
        return x
    if x == 0:
        return "0"
    s = f"{x:.{d}e}"
    m, e = s.split("e")
    return f"\\ensuremath{{{m}\\times 10^{{{int(e)}}}}}"


def fix(x, d=1):
    return "--" if x is None else f"{x:.{d}f}"


class Numbers:
    def __init__(self):
        self.rows = []

    def add(self, name, tex_value, plain, source, note=""):
        assert name.isalpha(), name
        self.rows.append((name, tex_value, plain, source, note))


def outputs(manifest, name, preset):
    for r in manifest["steps"]:
        if r["name"] == name and r["preset"] == preset and r["returncode"] == 0 and r["outputs"]:
            return os.path.join(ROOT, "results", r["outputs"][-1])
    return None


def main():
    man = json.load(open(os.path.join(ROOT, "results", "MANIFEST.json")))
    N = Numbers()
    for p, W in PW.items():
        rp = os.path.join(ROOT, "Optimization", "reference", f"reference_{p}.json")
        if os.path.exists(rp):
            ref = json.load(open(rp))
            src = os.path.relpath(rp, ROOT)
            for h in ref["horizons"]:
                N.add(f"QKD{W}RefHorizon{NXW[h['n_X']]}", fix(h.get("horizon_km"), 2), h.get("horizon_km"), src,
                      "km, largest L with l > 0 (bisection)")
        b = outputs(man, "benchmark", p)
        if b:
            s = json.load(open(os.path.join(b, "summary.json")))
            src = os.path.relpath(os.path.join(b, "summary.json"), ROOT)
            for m, M in (("new_cold", "NewCold"), ("new_warm", "NewWarm"), ("lookup_table", "Lookup"),
                         ("legacy_DA", "OldDA"), ("legacy_Hessian", "OldHessian"), ("legacy_Hessian_FoR", "OldFoR")):
                if m in s["regret"]:
                    a = s["regret"][m]["all"]
                    N.add(f"QKD{W}{M}PctWithin", fix(a["pct_within_1e6"]), a["pct_within_1e6"], src,
                          "% of reference points within 1e-6 relative")
                    N.add(f"QKD{W}{M}RegretMax", sci(a["max"]), a["max"], src, "max relative regret")
                    N.add(f"QKD{W}{M}RegretMedian", sci(a["median"]), a["median"], src, "median relative regret")
                if s.get("tail_regret") and m in s["tail_regret"]["regret"]:
                    a = s["tail_regret"]["regret"][m]["all"]
                    N.add(f"QKD{W}Tail{M}PctWithin", fix(a["pct_within_1e6"]), a["pct_within_1e6"], src,
                          "tail reference: % within 1e-6")
                    N.add(f"QKD{W}Tail{M}RegretMax", sci(a["max"]), a["max"], src, "tail reference: max regret")
            for k, v in (s.get("crosscheck") or {}).items():
                K = "Main" if k == "main" else "Tail"
                N.add(f"QKD{W}Xcheck{K}PctAgree", fix(v["pct_agree_1e6"]), v["pct_agree_1e6"], v["path"],
                      "independent SLSQP cross-check: % of points |d| <= 1e-6")
                N.add(f"QKD{W}Xcheck{K}Weak", str(v["n_reference_weakness"]), v["n_reference_weakness"], v["path"],
                      "points where the cross-check beats the reference")
            t = s["timing"]["new_warm"]
            N.add(f"QKD{W}SweepMsPerPoint", fix(t["steady_per_point_ms"], 2), t["steady_per_point_ms"], src,
                  "warm sweep steady ms/point (2400 points)")
            N.add(f"QKD{W}SweepCompileS", fix(t["compile_overhead_s"]), t["compile_overhead_s"], src, "s JIT compile")
            for nx in (1e4, 1e9):
                d = s["smoothness"][f"{nx:.0e}"]
                for m, M in (("new_warm", "NewWarm"), ("legacy_DA", "OldDA"), ("legacy_Hessian_FoR", "OldFoR")):
                    if m in d:
                        v = d[m]["S"]["TV_per_km"]
                        N.add(f"QKD{W}{M}TV{NXW[nx]}", sci(v), v, src, "TV/km on S (tau = 1e-3), events excluded")
        tdir = outputs(man, "timing", p)
        if tdir:
            tj = json.load(open(os.path.join(tdir, "timing.json")))
            src = os.path.relpath(os.path.join(tdir, "timing.json"), ROOT)
            da = [d["seconds"] for d in tj["legacy_DA"]]
            N.add(f"QKD{W}DASecMin", fix(min(da)), min(da), src, "s per point, DA (archived settings)")
            N.add(f"QKD{W}DASecMax", fix(max(da)), max(da), src, "s per point, DA (archived settings)")
            w, lt = tj["jax_warm_batch1"], tj["lookup_table_batch1"]
            N.add(f"QKD{W}WarmDeviceUs", fix(w["steady_device_us"]["median"], 0), w["steady_device_us"]["median"], src,
                  "us, batch-1 steady-state device-resident, median")
            N.add(f"QKD{W}WarmEndToEndUs", fix(w["end_to_end_us"]["median"], 0), w["end_to_end_us"]["median"], src,
                  "us, end-to-end, median")
            N.add(f"QKD{W}LookupDeviceUs", fix(lt["steady_device_us"]["median"]), lt["steady_device_us"]["median"],
                  src, "us, lookup table, device-resident, median")
            N.add(f"QKD{W}LookupEndToEndUs", fix(lt["end_to_end_us"]["median"]), lt["end_to_end_us"]["median"], src,
                  "us, lookup table, end-to-end, median")
        adir = outputs(man, "analyses", p)
        if adir:
            aj = json.load(open(os.path.join(adir, "analyses.json")))
            src = os.path.relpath(os.path.join(adir, "analyses.json"), ROOT)
            r = next((x for x in aj["robustness"] if x["L"] == 100.0 and x["n_X"] == 1e9 and x["alpha"] == 0.21), None)
            if r and r["regret"] is not None:
                N.add(f"QKD{W}DriftOldRatio", fix(r["old_retention_ratio"], 3), r["old_retention_ratio"], src,
                      "R(p_nom|alpha=0.21)/R(p_nom|alpha=0.2), L=100 km, n_X=1e9")
                N.add(f"QKD{W}DriftRegret", sci(r["regret"]), r["regret"], src,
                      "regret vs re-optimised at alpha=0.21, L=100 km, n_X=1e9")
    pd = outputs(man, "physics_diff", "all")
    if pd:
        pj = json.load(open(os.path.join(pd, "physics_diff.json")))
        src = os.path.relpath(os.path.join(pd, "physics_diff.json"), ROOT)
        for k, v in pj["targets"].items():
            W = PW.get(k)
            if not W:
                continue
            N.add(f"QKD{W}TargetROne", sci(v["T1_R0_1e9"], 3), v["T1_R0_1e9"], src, "R(0, 1e9), paper ~1e-2")
            N.add(f"QKD{W}TargetRatio", fix(v["T2_ratio"], 3) if v["T2_ratio"] else "--", v["T2_ratio"], src,
                  "R(100,1e9)/R(100,1e7), paper ~1.75")
            t3 = v["T3_L_at_R_1em7_nX1e4"]
            N.add(f"QKD{W}TargetReach", fix(t3, 1) if isinstance(t3, float) else "--", t3, src,
                  "km where R falls to 1e-7, n_X=1e4, paper ~135")

    tex = ["% AUTO-GENERATED by Optimization/report/make_numbers.py from results/MANIFEST.json -- DO NOT EDIT.",
           f"% {len(N.rows)} values. Sources: see results/NUMBERS.md."]
    for name, tv, _, _, _ in N.rows:
        tex.append(f"\\providecommand{{\\{name}}}{{}}\\renewcommand{{\\{name}}}{{{tv}}}")
    with open(os.path.join(ROOT, "result_report", "generated_numbers.tex"), "w") as fh:
        fh.write("\n".join(tex) + "\n")
    md = ["# Generated numbers (do not edit; `python Optimization/report/make_numbers.py`)", "",
          "| macro | value | meaning | source |", "|---|---|---|---|"]
    for name, _, pv, src, note in N.rows:
        pvs = f"{pv:.6g}" if isinstance(pv, float) else str(pv)
        md.append(f"| `\\{name}` | {pvs} | {note} | `{src}` |")
    with open(os.path.join(ROOT, "results", "NUMBERS.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print(f"[numbers] wrote {len(N.rows)} values")


if __name__ == "__main__":
    main()
