"""Pitch figure: optimal key rate vs distance, one line per session size (main .venv; reads labels only).

Sequential ink ramp (block size is ordered); each line directly labelled; maximum distance marked. Same verified
labels as params_by_blocksize.png. Writes qml/slides/assets/keyrate_simple.png.
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from qml import common as C  # noqa: E402

RAMP = ["#b9b8b3", "#9a9994", "#7a7975", "#5c5b57", "#353432", "#0b0b0b"]


def main():
    f = sorted(glob.glob(os.path.join(C.DATA, "labels_*.npz")))[0]
    d = dict(np.load(f, allow_pickle=False))
    meta = json.load(open(f.replace(".npz", ".json")))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 12})
    fig, ax = plt.subplots(figsize=(11, 6.2))
    for e, col in zip(range(4, 10), RAMP):
        m = (np.abs(np.log10(d["n_X"]) - e) < 1e-9) & d["on_lattice"] & (d["R"] > 0)
        o = np.argsort(d["L"][m]); L, R = d["L"][m][o], d["R"][m][o]
        h = float(d["horizon_label"][m][0])
        ax.plot(L, R, color=col, lw=2.4, label=f"session size 10^{e}  →  max {h:.0f} km")
        ax.plot([h], [1e-12], marker="v", ms=9, color=col, clip_on=False)
    h_, l_ = ax.get_legend_handles_labels()
    ax.legend(h_[::-1], l_[::-1], loc="lower left", frameon=False, fontsize=11)
    ax.set_yscale("log"); ax.set_ylim(1e-12, 3e-2); ax.set_xlim(0, 200)
    ax.set_xlabel("fiber length (km)"); ax.set_ylabel("secret key per laser pulse (best settings)")
    ax.grid(color="#e6e5e0", lw=0.8)
    ax.set_title("Longer fiber, less key: each session size has a maximum distance (▼)", loc="left", fontsize=13)
    fig.text(0.995, 0.005, f"labels {os.path.basename(f)} | phys {meta['stamp']['physics_source_hash']}",
             ha="right", fontsize=6, color="0.45")
    fig.tight_layout()
    out = os.path.join(C.QML, "slides", "assets", "keyrate_simple.png")
    fig.savefig(out, dpi=170); plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
