"""Regenerate every result in results/ from scripts, in order, and write results/MANIFEST.json.

Steps (presets: lim = default, lim_literal = printed eta_ch diagnostic, legacy = src/qkd physics):
  1. trusted reference (main grid)            all presets
  2. dense tail reference                     all presets
  3. independent solver cross-check (main, tail)  all presets
  4. physics differences + paper targets
  5. benchmark                                 all presets (reads 1-3)
  6. analyses (sensitivity, robustness, extrapolation)   lim, legacy
  7. timing (run last, machine otherwise idle) lim, legacy
  8. unit tests
  9. report numbers: result_report/generated_numbers.tex + results/NUMBERS.md

Usage: python Optimization/run_all.py [--from STEP]   (logs in results/logs/)
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
PRESETS = ["lim", "lim_literal", "legacy"]


def steps():
    s = []
    for p in PRESETS:
        s.append(("reference", p, ["Optimization/reference/build_reference.py", "--preset", p]))
    for p in PRESETS:
        s.append(("reference_tail", p, ["Optimization/reference/build_reference.py", "--preset", p, "--tail"]))
    for p in PRESETS:
        s.append(("crosscheck", p, ["Optimization/reference/crosscheck_scipy.py", "--preset", p]))
        s.append(("crosscheck_tail", p, ["Optimization/reference/crosscheck_scipy.py", "--preset", p, "--tail"]))
    s.append(("physics_diff", "all", ["Optimization/analysis/physics_diff.py"]))
    for p in PRESETS:
        s.append(("benchmark", p, ["Optimization/benchmark/run_benchmark.py", "--preset", p]))
    for p in ("lim", "legacy"):
        s.append(("analyses", p, ["Optimization/analysis/analyses.py", "--preset", p]))
    for p in ("lim", "legacy"):
        s.append(("timing", p, ["Optimization/benchmark/timing_benchmark.py", "--preset", p]))
    s.append(("unittest", "all", ["-m", "unittest", "discover", "tests"]))
    return s


def listing():
    d = os.path.join(ROOT, "results")
    return {x: os.path.getmtime(os.path.join(d, x)) for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(os.path.join(ROOT, "results", "logs"), exist_ok=True)
    mpath = os.path.join(ROOT, "results", "MANIFEST.json")
    manifest = json.load(open(mpath)) if (a.start and os.path.exists(mpath)) else dict(steps=[])
    env = dict(os.environ, JAX_PLATFORMS=os.environ.get("JAX_PLATFORMS", "cpu"))
    for k, (name, preset, args) in enumerate(steps()):
        if k < a.start:
            continue
        before = listing()
        log = os.path.join(ROOT, "results", "logs", f"{k:02d}_{name}_{preset}.log")
        t = time.perf_counter()
        print(f"[run_all] {k:02d} {name} {preset} ...", flush=True)
        with open(log, "w") as fh:
            rc = subprocess.run([PY, *args], cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
        after = listing()
        new = sorted(x for x in after if x not in before or after[x] > before.get(x, 0) + 1e-6)
        new = [x for x in new if x != "logs"]
        rec = dict(step=k, name=name, preset=preset, cmd=" ".join(args), returncode=rc,
                   seconds=round(time.perf_counter() - t, 1), log=os.path.relpath(log, ROOT), outputs=new)
        manifest["steps"] = [r for r in manifest["steps"] if r["step"] != k] + [rec]
        with open(mpath, "w") as fh:
            json.dump(manifest, fh, indent=1)
        print(f"[run_all]    rc={rc} {rec['seconds']}s -> {new}", flush=True)
        if rc != 0:
            print(f"[run_all] FAILED, see {rec['log']}; resume with --from {k}")
            sys.exit(rc)
    rc = subprocess.run([PY, "Optimization/report/make_numbers.py"], cwd=ROOT).returncode
    sys.exit(rc)


if __name__ == "__main__":
    main()
