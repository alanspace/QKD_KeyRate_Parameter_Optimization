"""Per-seed diagnostics of the RFF ridge (qml env; no physics import). Explains the seed-to-seed spread in RESULTS.md.

The ridge uses D2 = 14 random frequencies, each a random {-1, 0, 1} combination of the trained PQC input scalings of
that seed (models.sample_frequencies). This script recomputes, per seed and variant, from the saved parts (W, B):
frequency magnitudes, training and validation MSE in z, the validation MSE at every lambda of the grid, and the
condition number of F^T F; and joins the certified test_bulk median regret (primary certification).
Usage: JAX_PLATFORMS=cpu PYTHONPATH=. .venv-qml/bin/python qml/diag_rff_seeds.py
Output: results/qml_rffdiag_<git>_<cfg>/rff_seeds.json
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from qml import common as C  # noqa: E402


def main():
    from qml import train as T, models as M
    from qml.report import certifications
    mf, _ = certifications()
    A = json.load(open(mf))
    path, d = T.load()
    out = dict(config=dict(script="qml/diag_rff_seeds.py", labels=os.path.basename(path), certification=os.path.relpath(mf, ROOT),
                           lambdas=T.LAMBDAS), seeds={})
    for var in ("dense", "sparse"):
        tr, lo, hi, X, Z, Xv, Zv = T._variant_data(d, var)
        X, Z = np.asarray(X), np.asarray(Z)
        for s in range(5):
            f = os.path.join(C.DATA, "final_parts", f"{var}__RFF-ridge__s{s}.npz")
            z = np.load(f)
            meta = json.loads(str(z["meta"]))
            W, B = z["w__W"], z["w__B"]
            F, Fv = M.rff_features(X, W), M.rff_features(Xv, W)
            val_by_lam = [float(np.mean((np.clip(Fv @ M.fit_ridge(F, Z, lam), -1, 1) - Zv) ** 2)) for lam in T.LAMBDAS]
            out["seeds"][f"{var}__s{s}"] = dict(
                variant=var, seed=s, lam=meta.get("lam"), device=meta.get("device", {}).get("backend"),
                freq_abs_median=float(np.median(np.abs(W))), freq_abs_max_uL=float(np.abs(W[:, 0]).max()),
                freq_abs_max_v=float(np.abs(W[:, 1]).max()),
                train_mse_z=float(np.mean((F @ B - Z) ** 2)), val_mse_z=float(np.mean((np.clip(Fv @ B, -1, 1) - Zv) ** 2)),
                val_mse_z_by_lambda=val_by_lam, cond_FtF=float(np.linalg.cond(F.T @ F)),
                test_bulk_median_regret=A["keys"][f"{var}__RFF-ridge__s{s}"]["groups"]["test_bulk"]["k0"]["median"])
            r = out["seeds"][f"{var}__s{s}"]
            print(f"[rff] {var} s{s}: lam {r['lam']}, median|W| {r['freq_abs_median']:.2f}, train MSE {r['train_mse_z']:.2e}, "
                  f"val MSE {r['val_mse_z']:.2e}, test_bulk median {r['test_bulk_median_regret']:.2e}", flush=True)
    for var in ("dense", "sparse"):
        rs = [v for v in out["seeds"].values() if v["variant"] == var]
        a = np.argsort([r["train_mse_z"] for r in rs]); b = np.argsort([r["test_bulk_median_regret"] for r in rs])
        out[f"{var}_rank_train_mse_equals_rank_test_regret"] = bool(np.array_equal(a, b))
        out[f"{var}_spearman_train_mse_vs_test_regret"] = float(np.corrcoef(np.argsort(a), np.argsort(b))[0, 1])
    od = C.out_dir("rffdiag", out["config"])
    C.write_json(os.path.join(od, "rff_seeds.json"), out)
    print(f"[rff] wrote {os.path.relpath(od, ROOT)}", flush=True)


if __name__ == "__main__":
    main()
