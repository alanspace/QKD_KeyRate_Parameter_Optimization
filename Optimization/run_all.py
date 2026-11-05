"""Run result-producing scripts and record every run in results/MANIFEST.json.

Step ids are "<name>:<preset>" (preset "all" for preset-independent steps). Default: the full pipeline in
the order below. --only runs the given ids (in the given order); --parallel runs them concurrently (each
with its own log; the manifest is written once all have finished). Every record holds the command, return
code, wall time, log path, the result directories created or modified, and the power state at start.

  reference:<p> reference_tail:<p>                     (p in lim, lim_literal, legacy)
  crosscheck:<p> crosscheck_tail:<p>
  compare_references:all  local_opt_check:lim
  cliff_test:all  dense_scan:all  gamma_report:all  physics_diff:all
  benchmark:<p>  analyses:<p>  timing:<p>  make_figures:all  unittest:all  numbers:all

Usage: python Optimization/run_all.py [--only ID [ID ...]] [--parallel]
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
PY = sys.executable
PRESETS = ["lim", "lim_literal", "legacy"]


def command(sid):
    name, preset = sid.split(":")
    p = ["--preset", preset]
    table = {
        "reference": ["Optimization/reference/build_reference.py", *p],
        "reference_tail": ["Optimization/reference/build_reference.py", *p, "--tail"],
        "crosscheck": ["Optimization/reference/crosscheck_scipy.py", *p],
        "crosscheck_tail": ["Optimization/reference/crosscheck_scipy.py", *p, "--tail"],
        "compare_references": ["Optimization/analysis/compare_references.py", "--old-dir", "pre_expm1", "lim", "lim_literal"],
        "local_opt_check": ["Optimization/analysis/local_opt_check.py", *p],
        "cliff_test": ["Optimization/analysis/cliff_test.py"],
        "dense_scan": ["Optimization/analysis/dense_scan.py"],
        "gamma_report": ["Optimization/analysis/gamma_report.py"],
        "physics_diff": ["Optimization/analysis/physics_diff.py"],
        "benchmark": ["Optimization/benchmark/run_benchmark.py", *p],
        "analyses": ["Optimization/analysis/analyses.py", *p],
        "timing": ["Optimization/benchmark/timing_benchmark.py", *p],
        "make_figures": ["Optimization/report/make_figures.py"],
        "unittest": ["-m", "unittest", "discover", "tests"],
        "numbers": ["Optimization/report/make_numbers.py"],
    }
    return table[name]


def default_steps():
    s = [f"reference:{p}" for p in PRESETS] + [f"reference_tail:{p}" for p in PRESETS]
    s += [f"crosscheck:{p}" for p in PRESETS] + [f"crosscheck_tail:{p}" for p in PRESETS]
    s += ["physics_diff:all"] + [f"benchmark:{p}" for p in PRESETS]
    s += ["analyses:lim", "analyses:legacy", "timing:lim", "timing:legacy", "unittest:all"]
    return s


def listing():
    d = os.path.join(ROOT, "results")
    return {x: os.path.getmtime(os.path.join(d, x)) for x in os.listdir(d) if os.path.isdir(os.path.join(d, x))}


def power():
    try:
        from src.qkd_opt.provenance import power_state
        return power_state()
    except Exception as e:  # noqa: BLE001
        return dict(error=str(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--parallel", action="store_true")
    a = ap.parse_args()
    steps = a.only or default_steps()
    os.makedirs(os.path.join(ROOT, "results", "logs"), exist_ok=True)
    mpath = os.path.join(ROOT, "results", "MANIFEST.json")
    manifest = json.load(open(mpath)) if os.path.exists(mpath) else dict(steps=[])
    env = dict(os.environ, JAX_PLATFORMS=os.environ.get("JAX_PLATFORMS", "cpu"))
    stamp = time.strftime("%Y%m%d_%H%M%S")

    def record(sid, rc, secs, log, before, pw):
        after = listing()
        new = sorted(x for x in after if (x not in before or after[x] > before.get(x, 0) + 1e-6) and x != "logs")
        name, preset = sid.split(":")
        rec = dict(id=sid, name=name, preset=preset, cmd=" ".join(command(sid)), returncode=rc, seconds=round(secs, 1),
                   log=os.path.relpath(log, ROOT), outputs=new, power_at_start=pw, finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        manifest["steps"] = [r for r in manifest["steps"] if r.get("id") != sid] + [rec]
        return rec

    if a.parallel:
        before, pw, procs = listing(), power(), []
        for sid in steps:
            log = os.path.join(ROOT, "results", "logs", f"{stamp}_{sid.replace(':', '_')}.log")
            fh = open(log, "w")
            procs.append((sid, log, fh, time.perf_counter(),
                          subprocess.Popen([PY, *command(sid)], cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env)))
            print(f"[run_all] started {sid}", flush=True)
        recs = []
        for sid, log, fh, t, pr in procs:
            rc = pr.wait()
            fh.close()
            recs.append((sid, rc, time.perf_counter() - t, log))
        # outputs of parallel steps cannot be separated by mtime; attribute by directory-name prefix
        after = listing()
        new_all = sorted(x for x in after if (x not in before or after[x] > before.get(x, 0) + 1e-6) and x != "logs")
        for sid, rc, secs, log in recs:
            name = sid.split(":")[0]
            rec = record(sid, rc, secs, log, before, pw)
            rec["outputs"] = [x for x in new_all if x.startswith(name)]
            print(f"[run_all] {sid} rc={rc} {secs:.0f}s -> {rec['outputs']}", flush=True)
        with open(mpath, "w") as fh:
            json.dump(manifest, fh, indent=1)
        sys.exit(max(r[1] for r in recs))
    for sid in steps:
        before, pw = listing(), power()
        log = os.path.join(ROOT, "results", "logs", f"{stamp}_{sid.replace(':', '_')}.log")
        t = time.perf_counter()
        print(f"[run_all] {sid} ...", flush=True)
        with open(log, "w") as fh:
            rc = subprocess.run([PY, *command(sid)], cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
        rec = record(sid, rc, time.perf_counter() - t, log, before, pw)
        with open(mpath, "w") as fh:
            json.dump(manifest, fh, indent=1)
        print(f"[run_all]    rc={rc} {rec['seconds']}s -> {rec['outputs']}", flush=True)
        if rc != 0:
            print(f"[run_all] FAILED, see {rec['log']}")
            sys.exit(rc)


if __name__ == "__main__":
    main()
