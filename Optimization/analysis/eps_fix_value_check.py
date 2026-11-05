"""Step 1 check: the eps_sec custom_jvp fix must leave VALUES unchanged.
Evaluates preset lim (l, R, eps_sec) at 50 random interior points and 10 near-horizon points (the last two
R > 0 points per n_X of the archived pre-fix lim tail reference, first 10) with the code of a given commit.
Usage:  python eps_fix_value_check.py --eval <repo_dir> <out.npz>      (run with cwd = repo_dir)
        python eps_fix_value_check.py --compare <old.npz> <new.npz>
"""
import json
import os
import sys

import numpy as np


def points():
    rng = np.random.default_rng(424242)
    P, L, N = [], [], []
    for _ in range(50):
        P.append([rng.uniform(0.3, 0.7), rng.uniform(0.05, 0.2), rng.uniform(0.1, 0.6), rng.uniform(0.1, 0.35),
                  rng.uniform(0.5, 0.95)])
        L.append(rng.uniform(0.0, 120.0))
        N.append(float(10 ** rng.integers(4, 10)))
    here = os.path.dirname(os.path.abspath(__file__))
    ref = json.load(open(os.path.join(here, "..", "reference", "prefix_eps_fix", "reference_lim_tail.json")))
    near = []
    for nx in (1e4, 1e5, 1e6, 1e7, 1e8, 1e9):
        pts = sorted([p for p in ref["points"] if p["n_X"] == nx and p["R"] > 0], key=lambda p: p["L"])
        near += pts[-2:]
    for p in near[:10]:
        P.append(p["params"]); L.append(p["L"]); N.append(p["n_X"])
    return np.array(P), np.array(L), np.array(N)


if __name__ == "__main__":
    if sys.argv[1] == "--eval":
        repo, out = sys.argv[2], sys.argv[3]
        sys.path.insert(0, repo)
        import jax.numpy as jnp
        from src.qkd_opt import evaluate
        from src.qkd_opt.presets import get_preset
        lim = get_preset("lim")
        P, L, N = points()
        res = [evaluate(jnp.asarray(p), l_, n_, lim.consts, lim.flags) for p, l_, n_ in zip(P, L, N)]
        np.savez(out, l=[float(r["l"]) for r in res], R=[float(r["R"]) for r in res],
                 eps=[float(r["eps_sec"]) for r in res], L=L, N=N)
    else:
        a, b = np.load(sys.argv[2]), np.load(sys.argv[3])
        for k in ("l", "R", "eps"):
            rel = np.abs(a[k] - b[k]) / np.maximum(np.abs(a[k]), 1e-300)
            print(f"{k}: max relative difference {rel.max():.3e} (interior 50: {rel[:50].max():.3e}; "
                  f"near-horizon 10: {rel[50:].max():.3e}); argmax point L={a['L'][rel.argmax()]:.3f} n_X={a['N'][rel.argmax()]:.0e}")
