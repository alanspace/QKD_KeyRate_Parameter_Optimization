"""Slide assets (main .venv; no physics evaluation): copy report figures with provenance, draw the label-dataset figure.

Writes qml/slides/assets/{fig*.png, dataset_splits.png, SOURCES.md}. Every copied file is listed with its source
path and SHA-256 so the deck can be checked against the repository.
Palette: reference categorical slots 1-3 (blue, orange, aqua), validated all-pairs in the dataviz reference palette.
"""
import glob
import json
import os
import shutil
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from qml import common as C  # noqa: E402

A = os.path.join(C.QML, "slides", "assets")
FIGDIR = os.path.join(ROOT, "results", "figures", "900d9f88a8_58c9f38705cd")
COPY = ["fig1_keyrate.png", "fig2_presets.png", "fig3_horizon.png", "fig4_params_nx1e9.png", "fig5_tail.png",
        "fig6_regret_cdf.png", "fig7_ablation.png", "fig8_timing.png"]
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


def dataset_figure(path_npz, path_json, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = dict(np.load(path_npz, allow_pickle=False))
    lab = json.load(open(path_json))
    hz = {round(h["n_X"]): h["horizon_km"] for h in lab["horizons"]}
    xo = lab["crossovers"]
    train = [10.0 ** e for e in C.TRAIN_LOG10NX]
    fig = plt.figure(figsize=(13, 6.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.2, 1], height_ratios=[1, 1], wspace=0.25, hspace=0.55)
    ax = fig.add_subplot(gs[:, 0])
    order = sorted(hz)
    ypos = {nx: i for i, nx in enumerate(order)}          # evenly spaced rows: 3e6 and 10^6.5 stay distinct
    for nx, h in sorted(hz.items()):
        col = BLUE if any(abs(np.log10(nx) - np.log10(t)) < 1e-3 for t in train) else (ORANGE if nx in (3e6, 5e8) else AQUA)
        y = ypos[nx]
        ax.plot([0, h - 15], [y, y], color=col, lw=2.2, solid_capstyle="round")
        ax.plot([h - 15, h], [y, y], color=col, lw=6.5, solid_capstyle="butt", alpha=0.85)
        ax.plot([h], [y], marker="|", ms=11, color=INK, mew=1.5)
    for c in xo:
        ax.plot([c["crossover_km"]], [ypos[round(c["n_X"])]], marker="x", ms=7, mew=1.6, color=INK, zorder=5)
    ax.set_xlim(0, 200)
    ax.set_xlabel("fiber length L (km)", color=INK2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{nx:.2g}" for nx in order], fontsize=8)
    ax.set_ylabel("block size n_X (rows evenly spaced)", color=INK2)
    ax.grid(axis="x", color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=BLUE, lw=3, label="training block sizes (11; train / val / test interleaved)"),
               Line2D([], [], color=ORANGE, lw=3, label="held-out block sizes (3e6, 5e8)"),
               Line2D([], [], color=AQUA, lw=3, label="extrapolation (5e3, 2e9)"),
               Line2D([], [], color=INK2, lw=6.5, alpha=0.5, label="tail: last 15 km before the horizon (scored separately)"),
               Line2D([], [], color=INK, marker="|", ls="", ms=11, label="label horizon (largest L with key > 0)"),
               Line2D([], [], color=INK, marker="x", ls="", ms=7, label="below-horizon branch switch (13; +-1 km reported separately)")]
    ax.legend(handles=handles, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.13), ncol=2, frameon=False)
    ax.set_title(f"{lab['n_points']} certified labels (one warm sweep, 15 block sizes, preset lim)", fontsize=10, loc="left")

    def zoom(axz, nx, lo, hi, title):
        m = (d["n_X"] == nx) & (d["L"] >= lo) & (d["L"] <= hi) & d["on_lattice"]
        for s, kw in (("train", dict(marker="o", ms=7, color=INK, ls="")),
                      ("val", dict(marker="o", ms=7, mfc="white", mec=INK, ls="")),
                      ("test", dict(marker="o", ms=3.5, color=INK2, ls=""))):
            mm = m & (d["split"] == s)
            axz.plot(d["L"][mm], np.zeros(mm.sum()), label=s, **kw)
        axz.set_yticks([]); axz.set_xlim(lo - 0.02 * (hi - lo), hi + 0.02 * (hi - lo))
        axz.set_title(title, fontsize=9, loc="left")
        for s in ("top", "right", "left"):
            axz.spines[s].set_visible(False)
        axz.set_xlabel("L (km)", fontsize=8, color=INK2)
    z1 = fig.add_subplot(gs[0, 1]); zoom(z1, 1e4, 50.0, 52.0, "bulk, n_X = 1e4: train every 0.5 km")
    z2 = fig.add_subplot(gs[1, 1]); zoom(z2, 1e4, 125.0, 125.5, "tail, n_X = 1e4: train every 0.125 km")
    z1.legend(fontsize=8, ncol=3, frameon=False, loc="upper left", bbox_to_anchor=(0, 1.0))
    st = lab["stamp"]
    fig.text(0.995, 0.005, f"labels git {st['git_commit']} | cfg {st['config_hash']} | phys {st['physics_source_hash']}",
             ha="right", fontsize=6, color="0.45")
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(A, exist_ok=True)
    lines = ["# Slide assets: sources", "", "| file | source | SHA-256 (first 16) |", "|---|---|---|"]
    for f in COPY:
        src = os.path.join(FIGDIR, f)
        shutil.copy2(src, os.path.join(A, f))
        lines.append(f"| {f} | `{os.path.relpath(src, ROOT)}` (make_figures.py, git 900d9f88a8) | {C.sha256(src)[:16]} |")
    npz = sorted(glob.glob(os.path.join(C.DATA, "labels_*.npz")))[-1]
    js = npz.replace(".npz", ".json")
    out = os.path.join(A, "dataset_splits.png")
    dataset_figure(npz, js, out)
    lines.append(f"| dataset_splits.png | drawn by `qml/slides/make_assets.py` from `{os.path.relpath(npz, ROOT)}` | {C.sha256(out)[:16]} |")
    lines.append("| pqc_circuit.png, pqc_circuit.txt | drawn by `qml/slides/draw_circuit.py` (.venv-qml, PennyLane) | see file |")
    with open(os.path.join(A, "SOURCES.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("assets written:", sorted(os.listdir(A)))


if __name__ == "__main__":
    main()
