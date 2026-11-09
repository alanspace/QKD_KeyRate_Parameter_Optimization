"""All five optimal settings across distance, one panel per block size (main .venv; reads labels only).

Writes qml/slides/assets/params_by_blocksize.png. Parameter colours: violet, magenta, green, yellow, red (validated as a
5-slot categorical set on the light surface; two slots below 3:1 contrast, so every line is directly labelled).
Use --labels to draw from repaired labels (qml/data/repaired/...) once Amendment 4 has run.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from qml import common as C  # noqa: E402

COLS = ["#6250d6", "#e87ba4", "#008300", "#eda100", "#e34948"]
NAMES = ["μ₁ signal intensity", "μ₂ decoy intensity", "P(μ₁) signal prob.", "P(μ₂) decoy prob.", "q_x basis prob."]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=None)
    a = ap.parse_args()
    f = a.labels or sorted(glob.glob(os.path.join(C.DATA, "labels_*.npz")))[0]
    d = dict(np.load(f, allow_pickle=False))
    meta = json.load(open(f.replace(".npz", ".json")))
    from qml.repair_labels import isolated_jumps          # every visible jump, logged by the sweep or not
    xo = [dict(n_X=j["n_X"], km=j["km"]) for j in isolated_jumps(d, np.ones(len(d["L"]), bool))]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.6), sharey=True)
    for ax, e in zip(axes.flat, range(4, 10)):
        nx = 10.0 ** e
        m = (np.abs(np.log10(d["n_X"]) - e) < 1e-9) & d["on_lattice"] & (d["R"] > 0)
        o = np.argsort(d["L"][m])
        L, P = d["L"][m][o], d["params"][m][o]
        h = float(d["horizon_label"][m][0])
        for i in range(5):
            ax.plot(L, P[:, i], color=COLS[i], lw=2.0)
        ax.axvline(h, color=INK2, lw=0.8, ls=":")
        ax.text(h - 2, 0.985, f"max distance\n{h:.0f} km", ha="right", va="top", fontsize=8, color=INK2)
        for c in xo:
            if abs(np.log10(c["n_X"]) - e) < 1e-6:
                ax.plot([c["km"]], [-0.02], marker="^", ms=6, color=INK, clip_on=False)
        ax.set_xlim(0, 200); ax.set_ylim(-0.03, 1.0)
        ax.set_title(f"block size n_X = 10^{e}", loc="left", fontsize=11)
        ax.grid(color=GRID, lw=0.8)
        if e >= 7:
            ax.set_xlabel("fiber length (km)", color=INK2)
    # direct labels on the first panel, at the left end of each curve
    ax0 = axes.flat[5]
    m = (np.abs(np.log10(d["n_X"]) - 9) < 1e-9) & d["on_lattice"] & (d["R"] > 0)
    o = np.argsort(d["L"][m]); P = d["params"][m][o]
    ys = P[0].copy()                                       # spread label heights so no two are closer than 0.07
    order = np.argsort(ys)
    pos = ys[order] + 0.03
    for k in range(1, 5):
        pos[k] = max(pos[k], pos[k - 1] + 0.07)
    for k, i in enumerate(order):
        ax0.text(3, pos[k], NAMES[i], color=INK, fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=COLS[i], lw=1.2))
    for ax in axes[:, 0]:
        ax.set_ylabel("optimal value", color=INK2)
    from matplotlib.lines import Line2D
    hl = [Line2D([], [], color=COLS[i], lw=2.5, label=NAMES[i]) for i in range(5)]
    hl.append(Line2D([], [], color=INK, marker="^", ls="", label="optimum switches between two nearly equal settings (key rate unaffected)"))
    fig.legend(handles=hl, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("The five optimal settings across distance: flat at short and medium range, changing fast near the maximum distance",
                 x=0.01, ha="left", fontsize=12)
    st = meta["stamp"]
    fig.text(0.995, 0.995, f"labels {os.path.basename(f)} | phys {st['physics_source_hash']}", ha="right", va="top", fontsize=6, color="0.45")
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    out = os.path.join(C.QML, "slides", "assets", "params_by_blocksize.png")
    fig.savefig(out, dpi=160); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
