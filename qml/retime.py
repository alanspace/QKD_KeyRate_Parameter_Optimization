"""Dedicated batch-1 re-timing on a quiet machine (main .venv). Supersedes the batch-1 timings recorded inside the two
certify --final runs, which differed by up to ~2x because each was a single pass taken in the middle of a heavy job.

Same measurement code and the same points as certify --final (qml/certify.py time_kstep / time_cold / time_warm; dense
PQC-full seed-0 predictions; 35 test_bulk + 15 test_tail points, rng seed 1; cold on the first 20), but:
  - refuses unless on AC power and the machine is idle: 1-min load average <= MAX_LOAD after a settle period, and no
    other qml / benchmark Python process running (the load and the top CPU processes are recorded);
  - repeats the whole measurement N_PASSES times and reports the median over passes and the spread (min..max).
Output: results/qml_retime_<git>_<cfg>/retime.json
Usage: .venv/bin/python qml/retime.py
"""
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from qml import common as C  # noqa: E402

N_PASSES = 5
MAX_LOAD = 2.0
SETTLE_S = 60
BUSY = ("qml/certify.py", "qml/repair_labels.py", "qml/train.py", "qml/check_", "timing_benchmark.py", "da_dataset.py")


def idle_check():
    me = os.getpid()
    ps = subprocess.run(["ps", "-axo", "pid=,pcpu=,command="], capture_output=True, text=True).stdout.splitlines()
    rows = []
    for line in ps:
        parts = line.split(None, 2)
        if len(parts) == 3:
            rows.append((int(parts[0]), float(parts[1]), parts[2]))
    busy = [r for r in rows if r[0] != me and "python" in r[2].lower() and any(b in r[2] for b in BUSY)]
    top = sorted(rows, key=lambda r: -r[1])[:6]
    return dict(load_1_5_15=list(os.getloadavg()), busy=[r[2][:160] for r in busy],
                top_cpu=[dict(pcpu=r[1], command=os.path.basename(r[2].split()[0])) for r in top])


def main():
    pw = C.require_ac("retime")
    time.sleep(SETTLE_S)
    pre_state = idle_check()
    if pre_state["busy"]:
        sys.exit(f"retime: refusing, other jobs running: {pre_state['busy']}")
    if pre_state["load_1_5_15"][0] > MAX_LOAD:
        sys.exit(f"retime: refusing, 1-min load average {pre_state['load_1_5_15'][0]:.2f} > {MAX_LOAD}")
    from qml import certify as K
    from qml.report import certifications, repaired_labels
    from src.qkd_opt.provenance import stamp, to_jsonable
    d = dict(np.load(repaired_labels(), allow_pickle=False))
    mf, _ = certifications()
    M = json.load(open(mf))
    P = dict(np.load(os.path.join(os.path.dirname(mf), "perpoint.npz")))
    ev = P["ev_idx"]
    pq = os.path.join(ROOT, M["config"]["predictions"])
    Q = dict(np.load(pq)); idxq = Q.pop("idx")
    pos_in_q = {int(i): k for k, i in enumerate(idxq)}
    evq = np.array([pos_in_q[int(i)] for i in ev])
    G = K.groups(d)
    rng = np.random.default_rng(1)                      # the same pick as certify --final
    tb = np.where(G["test_bulk"][ev])[0]
    tt = np.where(G["test_tail"][ev])[0]
    pick = np.concatenate([rng.choice(tb, min(35, tb.size), replace=False), rng.choice(tt, min(15, tt.size), replace=False)])
    u_pick = Q["dense__PQC-full__s0"][evq[pick]]
    cfg = dict(script="qml/retime.py", certification=os.path.relpath(mf, ROOT), predictions=M["config"]["predictions"],
               n_passes=N_PASSES, max_load=MAX_LOAD, settle_s=SETTLE_S, n_kstep=int(pick.size), n_cold=20,
               n_warm=int(pick.size), pick="certify --final: rng seed 1, 35 test_bulk + 15 test_tail")
    st = stamp(cfg)
    passes = []
    for p_ in range(N_PASSES):
        t0 = time.perf_counter()
        rec = dict(kstep=[K.time_kstep(d, ev[pick], u_pick, k) for k in (1, 2, 3)],
                   cold=K.time_cold(d, ev[pick[:20]]), warm=K.time_warm(d, ev[pick]),
                   load_1min_after=os.getloadavg()[0])
        rec["pass_s"] = time.perf_counter() - t0
        passes.append(rec)
        print(f"[retime] pass {p_ + 1}/{N_PASSES} ({rec['pass_s']:.0f}s, load {rec['load_1min_after']:.2f}): "
              + ", ".join(f"k={t['k']} {t['end_to_end_us_median']:.0f}" for t in rec["kstep"])
              + f", cold {rec['cold']['end_to_end_us_median']:.0f}, warm {rec['warm']['end_to_end_us_median']:.0f} µs", flush=True)

    def summ(vals):
        v = np.asarray(vals, float)
        return dict(median=float(np.median(v)), min=float(v.min()), max=float(v.max()), passes=v.tolist())
    summary = dict(
        kstep=[dict(k=k, end_to_end_us=summ([p["kstep"][k - 1]["end_to_end_us_median"] for p in passes]),
                    device_us=summ([p["kstep"][k - 1]["device_us_median"] for p in passes]),
                    iters_median=passes[0]["kstep"][k - 1]["iters_median"]) for k in (1, 2, 3)],
        cold=dict(end_to_end_us=summ([p["cold"]["end_to_end_us_median"] for p in passes]),
                  p90_us=summ([p["cold"]["end_to_end_us_p90"] for p in passes]), settings=passes[0]["cold"]["settings"]),
        warm=dict(end_to_end_us=summ([p["warm"]["end_to_end_us_median"] for p in passes]),
                  iters_median=passes[0]["warm"]["iters_median"], settings=passes[0]["warm"]["settings"]))
    # same layout as certify's metrics["timing"], with the median over passes, so report.py can use it directly
    timing = dict(batch1_kstep=[dict(k=s["k"], n=int(pick.size), end_to_end_us_median=s["end_to_end_us"]["median"],
                                     device_us_median=s["device_us"]["median"], iters_median=s["iters_median"])
                                for s in summary["kstep"]],
                  batch1_cold=dict(n=20, end_to_end_us_median=summary["cold"]["end_to_end_us"]["median"],
                                   end_to_end_us_p90=summary["cold"]["p90_us"]["median"], settings=summary["cold"]["settings"]),
                  batch1_warm=dict(n=int(pick.size), end_to_end_us_median=summary["warm"]["end_to_end_us"]["median"],
                                   iters_median=summary["warm"]["iters_median"], settings=summary["warm"]["settings"]))
    res = dict(stamp=st, config=cfg, power_at_start=pw, idle_before=pre_state, idle_after=idle_check(),
               summary=summary, timing=timing, passes=passes)
    od = C.out_dir("retime", cfg)
    C.write_json(os.path.join(od, "retime.json"), to_jsonable(res))
    print(f"[retime] wrote {os.path.relpath(od, ROOT)}", flush=True)


if __name__ == "__main__":
    main()
