"""Pitch-deck showcase figures (main .venv): the five optimal parameters and the key rate across distance.

  params_across_distance.png    optimal parameters (labels) for three block sizes; no model needed
  showcase_predictions.png      (when Phase 4 results exist) optimal vs predicted parameters and the fraction of the
                                optimal key rate the prediction delivers, 1 - regret, at test points of n_X = 1e6
                                and at the held-out n_X = 5e8; models: PQC-full, NN-tuned, lookup-2D (seed 0)

Colours (dataviz reference slots 1-3, fixed meaning in the pitch deck): quantum circuit = blue, classical NN = orange,
lookup table = aqua; the optimum (labels) is ink.
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from qml import common as C  # noqa: E402

A = os.path.join(C.QML, "slides", "assets")
BLUE, ORANGE, AQUA, INK, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e6e5e0"
NAMES = ["μ₁ (signal)", "μ₂ (decoy)", "P(μ₁)", "P(μ₂)", "q_x (basis)"]


def plt_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.8, "axes.edgecolor": INK2, "font.size": 10})
    return plt


def labels(path=None):
    """The repaired labels (Amendment 4) when they exist, else the original labels; --labels overrides."""
    rep = sorted(glob.glob(os.path.join(C.DATA, "repaired", "labels_repaired_*.npz")), key=os.path.getmtime)
    f = path or (rep[-1] if rep else sorted(glob.glob(os.path.join(C.DATA, "labels_*.npz")))[-1])
    return f, dict(np.load(f, allow_pickle=False))


def optimal_only(d, out):
    plt = plt_setup()
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4))
    shades = {1e4: "#9a9994", 1e6: "#52514e", 1e9: INK}
    for nx, col in shades.items():
        m = (d["n_X"] == nx) & d["on_lattice"] & (d["R"] > 0)
        o = np.argsort(d["L"][m])
        L, P, R = d["L"][m][o], d["params"][m][o], d["R"][m][o]
        for i in range(5):
            axes.flat[i].plot(L, P[:, i], color=col, lw=1.8, label=f"n_X = {nx:.0e}")
        axes.flat[5].plot(L, R, color=col, lw=1.8)
    for i in range(5):
        axes.flat[i].set_title(NAMES[i], loc="left", fontsize=11)
    axes.flat[5].set_yscale("log"); axes.flat[5].set_title("key rate R at the optimum (bits/pulse)", loc="left", fontsize=11)
    for ax in axes[1]:
        ax.set_xlabel("fiber length L (km)")
    axes.flat[0].legend(frameon=False, fontsize=9)
    fig.suptitle("What has to be learned: the five optimal settings change with distance and block size", x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(out, dpi=170); plt.close(fig)


def with_predictions(d, out):
    try:
        pq = sorted(glob.glob(os.path.join(C.RESULTS, "qml_final_*", "predictions_query.npz")), key=os.path.getmtime)[-1]
        from qml.report import certifications
        mf = certifications()[0]   # certification against the repaired labels (Amendment 4)
    except (IndexError, SystemExit):
        print("no Phase 4 results yet: showcase_predictions.png not drawn")
        return False
    P = dict(np.load(pq)); idxq = P.pop("idx")
    pp = dict(np.load(os.path.join(os.path.dirname(mf), "perpoint.npz"))); ev = pp["ev_idx"]
    sys.path.insert(0, ROOT)
    import jax
    jax.config.update("jax_enable_x64", True)
    from src.qkd_opt import reparam
    plt = plt_setup()
    models = [("dense__PQC-full__s0", "quantum circuit (PQC)", BLUE), ("dense__NN-tuned__s0", "classical NN", ORANGE),
              ("dense__lookup-2D__s0", "lookup table", AQUA)]
    fig, axes = plt.subplots(2, 6, figsize=(17, 6.2), sharex="row")
    for row, (nx, split, title) in enumerate(((1e6, "test", "n_X = 1e6 (test distances)"),
                                              (5e8, "heldout", "n_X = 5e8 (block size never seen in training)"))):
        sel = np.where(np.isin(ev, np.where((d["n_X"] == nx) & (d["split"] == split) & d["scored"])[0]))[0]
        gi = ev[sel]
        o = np.argsort(d["L"][gi]); gi, sel = gi[o], sel[o]
        L = d["L"][gi]
        pos = {int(j): k for k, j in enumerate(idxq)}
        q = np.array([pos[int(j)] for j in gi])
        for i in range(5):
            ax = axes[row, i]
            ax.plot(L, d["params"][gi][:, i], color=INK, lw=2.4, label="optimum (verified solver)")
            for key, lab, col in models:
                p = np.asarray(reparam.params_of_u(P[key][q], 2e-4))
                ax.plot(L, p[:, i], color=col, lw=1.3, label=lab)
            if row == 0:
                ax.set_title(NAMES[i], loc="left", fontsize=11)
        ax = axes[row, 5]
        for key, lab, col in models:
            r = pp[f"{key}__k0"][sel]
            ax.plot(L, 100 * (1 - np.clip(r, 0, 1)), color=col, lw=1.3)
        ax.set_ylim(0, 101); ax.set_title("% of optimal key rate delivered", loc="left", fontsize=11) if row == 0 else None
        axes[row, 0].set_ylabel(title, fontsize=9)
        for a in axes[row]:
            a.set_xlabel("L (km)") if row == 1 else None
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
    st = json.load(open(mf))["stamp"]
    fig.text(0.995, 0.005, f"git {st['git_commit']} | {os.path.relpath(mf, ROOT)}", ha="right", fontsize=6, color="0.45")
    fig.suptitle("Predicted vs optimal settings across distance (seed 0); right: key rate the prediction delivers", x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)
    return True


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=None)
    f, d = labels(ap.parse_args().labels)
    print(f"labels: {os.path.relpath(f, ROOT)}", flush=True)
    os.makedirs(A, exist_ok=True)
    optimal_only(d, os.path.join(A, "params_across_distance.png"))
    drawn = with_predictions(d, os.path.join(A, "showcase_predictions.png"))
    print("params_across_distance.png written;", "showcase_predictions.png written" if drawn else "")


if __name__ == "__main__":
    main()
