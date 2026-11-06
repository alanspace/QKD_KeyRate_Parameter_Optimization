"""Run the qml study's steps and record every run in results/MANIFEST_qml.json (pattern of Optimization/run_all.py).

Each step runs in its own environment:
  main .venv     physics: labels, certification, hybrid Newton steps, solver timing
  .venv-qml      models: training, prediction, model latency (no physics import)
Every record holds the command, environment, return code, wall time, log path, the result paths created or
modified, and the power state at start (src/qkd_opt/provenance.power_state, loaded by file path).

Steps:
  labels           main  qml/labels.py                       (refuses unless on AC)
  pilot_train      qml   qml/pilot_train.py                  one training step per model configuration
  pilot_certify    main  qml/certify.py --pilot              1000 predictions: R, k = 1..3 Newton steps; batch-1
                                                             cold / warm solver timing

Usage: python qml/run_qml.py --only STEP [STEP ...]
"""
import argparse
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from qml import common as C  # noqa: E402

PY = {"main": os.path.join(ROOT, ".venv", "bin", "python"), "qml": os.path.join(ROOT, ".venv-qml", "bin", "python")}
STEPS = {
    "labels": ("main", ["qml/labels.py"]),
    "pilot_train": ("qml", ["qml/pilot_train.py"]),
    "pilot_certify": ("main", ["qml/certify.py", "--pilot"]),
}
WATCH = [os.path.join(ROOT, "results"), os.path.join(ROOT, "qml", "data")]


def listing():
    out = {}
    for d in WATCH:
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            p = os.path.join(d, name)
            out[os.path.relpath(p, ROOT)] = os.path.getmtime(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+", required=True, choices=sorted(STEPS))
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="extra args passed to the step")
    a = ap.parse_args()
    mpath = os.path.join(ROOT, "results", "MANIFEST_qml.json")
    manifest = json.load(open(mpath)) if os.path.exists(mpath) else dict(runs=[])
    logdir = os.path.join(ROOT, "results", "logs_qml")
    os.makedirs(logdir, exist_ok=True)
    rc_all = 0
    for sid in a.only:
        env_name, cmd = STEPS[sid]
        full = [PY[env_name], *cmd, *a.extra]
        log = os.path.join(logdir, f"{sid}_{time.strftime('%Y%m%d_%H%M%S')}.log")
        before, pw = listing(), C.power_state()
        t0 = time.time()
        with open(log, "w") as fh:
            rc = subprocess.call(full, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                                 env={**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": ROOT})
        secs = time.time() - t0
        after = listing()
        new = sorted(k for k, v in after.items() if before.get(k) != v)
        rec = dict(step=sid, env=env_name, command=[os.path.relpath(x, ROOT) if x.startswith(ROOT) else x for x in full],
                   returncode=rc, wall_s=round(secs, 1), log=os.path.relpath(log, ROOT), outputs=new,
                   power_at_start=pw, git=C.git("rev-parse", "--short=10", "HEAD"),
                   finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        manifest["runs"].append(rec)
        C.write_json(mpath, manifest)
        print(f"[run_qml] {sid}: rc={rc} {secs:.0f}s log={rec['log']}", flush=True)
        rc_all = rc_all or rc
    sys.exit(rc_all)


if __name__ == "__main__":
    main()
