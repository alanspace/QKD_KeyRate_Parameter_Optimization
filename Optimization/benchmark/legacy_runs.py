"""Frozen legacy optimizer outputs (untracked, gitignored JSONs) used as comparison data only.

Each file is checked against the SHA-256 recorded in Optimization/reference/legacy_da/PROVENANCE.md.
Key rates are RE-EVALUATED from the stored parameters with the legacy preset (the stored
'key_rate' of Hessian_FoR includes its Tikhonov penalty, REVIEW.md H4).
"""
import hashlib
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "Optimization", "run_scripts", "outputs", "JAX")
RUNS = {
    "legacy_DA": ("Pure_DA/run_PURE_DA_20260216_003022/qkd_grouped_dataset_20260216_003022.json",
                  "59ba882a1abb3a1e7fa2d3ec4362f0ef997756e55a8c6b0068569abc93561b8e"),
    "legacy_Hessian": ("Hessian/run_HESSIAN_20260216_142753/qkd_grouped_dataset_20260216_142753.json",
                       "853e56593334bfdb73e0244ffc5f297c0ece23c80c2563c153293d854bbda524"),
    "legacy_Hessian_FoR": ("Hessian_FoR/run_HESSIAN_FoR_20260217_014536/qkd_grouped_dataset_20260217_014536.json",
                           "5eed6d3fd23b9cf6995fcc99978c33aab036f3cc76d7e9f5f9df8aad1555ff53"),
}
NX = [1e4, 1e5, 1e6, 1e7, 1e8, 1e9]
GRID = np.linspace(0, 200, 400)
KEYS = ["mu_1", "mu_2", "P_mu_1", "P_mu_2", "P_X_value"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(name, check=True):
    """Return dict(L [400], params [6,400,5], R_stored [6,400], sha256)."""
    rel, expected = RUNS[name]
    path = os.path.join(OUT, rel)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name}: {path} missing (legacy outputs are untracked)")
    digest = sha256(path)
    if check and digest != expected:
        raise ValueError(f"{name}: checksum {digest[:12]} != recorded {expected}")
    d = json.load(open(path))
    P = np.zeros((len(NX), len(GRID), 5))
    Rs = np.zeros((len(NX), len(GRID)))
    for i, nx in enumerate(NX):
        e = sorted(d[str(float(nx))], key=lambda x: x["fiber_length"])
        L = np.array([x["fiber_length"] for x in e])
        assert np.allclose(L, GRID), name
        P[i] = [[x["optimized_params"][k] for k in KEYS] for x in e]
        Rs[i] = [x["key_rate"] for x in e]
    return dict(name=name, path=os.path.relpath(path, ROOT), sha256=digest, L=GRID, params=P, R_stored=Rs)


def reevaluate(run, preset):
    """R and l of the stored params under `preset` (legacy physics for these runs)."""
    import jax
    import jax.numpy as jnp
    from src.qkd_opt.objective import evaluate
    f = jax.jit(jax.vmap(lambda p, L, n: evaluate(p, L, n, preset.consts, preset.flags)),
                static_argnums=())
    P = run["params"].reshape(-1, 5)
    L = np.tile(run["L"], len(NX))
    n = np.repeat(NX, len(run["L"]))
    out = f(jnp.asarray(P), jnp.asarray(L), jnp.asarray(n))
    R = np.asarray(out["R"]).reshape(len(NX), -1)
    l = np.asarray(out["l"]).reshape(len(NX), -1)
    return R, l
