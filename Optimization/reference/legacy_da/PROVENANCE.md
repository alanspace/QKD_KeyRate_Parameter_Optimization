# Legacy optimizer outputs used as frozen comparison data

These files are **not tracked** (`*.json` is gitignored). They are used only as comparison data, never as
ground truth; see REVIEW.md C3. `Optimization/benchmark/legacy_runs.py` verifies these SHA-256 checksums
before using them and re-evaluates R from the stored parameters with the `legacy` preset.

| name | path (under `Optimization/run_scripts/outputs/JAX/`) | SHA-256 | bytes |
|---|---|---|---|
| legacy_DA | `Pure_DA/run_PURE_DA_20260216_003022/qkd_grouped_dataset_20260216_003022.json` | `59ba882a1abb3a1e7fa2d3ec4362f0ef997756e55a8c6b0068569abc93561b8e` | 1590788 |
| legacy_Hessian | `Hessian/run_HESSIAN_20260216_142753/qkd_grouped_dataset_20260216_142753.json` | `853e56593334bfdb73e0244ffc5f297c0ece23c80c2563c153293d854bbda524` | 1614772 |
| legacy_Hessian_FoR | `Hessian_FoR/run_HESSIAN_FoR_20260217_014536/qkd_grouped_dataset_20260217_014536.json` | `5eed6d3fd23b9cf6995fcc99978c33aab036f3cc76d7e9f5f9df8aad1555ff53` | 1614274 |

Known defects:
- The DA run has R = 0 for all L ≤ 100 km at n_X = 1e5–1e7. Below 100 km it is L-BFGS-B, not DA. It is unseeded and has no metadata, and the archived script cannot have produced it.
- The stored `key_rate` of Hessian_FoR includes its Tikhonov penalty.
