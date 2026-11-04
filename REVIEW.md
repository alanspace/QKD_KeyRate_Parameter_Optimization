# REVIEW — Optimization layer, finite-key decoy BB84 (Lim et al. 2014)

Phase 0, read-only review. No code was changed. Every number below was computed from the
result JSONs on disk with the unmodified `src/qkd` physics (throw-away scripts in
`/tmp/qkd_review/`, not committed). Paths are relative to the repo root.
`JAX_variants/` = `Optimization/run_scripts/JAX_variants/`; `outputs/JAX/` = `Optimization/run_scripts/outputs/JAX/`.

Runs analysed (the latest of each type):

| tag | file | grid |
|---|---|---|
| DA | `outputs/JAX/Pure_DA/run_PURE_DA_20260216_003022` | 400 pts |
| H | `outputs/JAX/Hessian/run_HESSIAN_20260216_142753` | 400 |
| FoR | `outputs/JAX/Hessian_FoR/run_HESSIAN_FoR_20260217_014536` | 400 |
| Hyb | `outputs/JAX/Hybrid/run_HYBRID_20260215_230846` | 400 |
| Base, Ultra | `outputs/JAX/Baseline/run_BASELINE_20260215_231330`, `outputs/JAX/Ultra/...` | 1000 |

---

## Summary (ranked)

| # | Sev. | Finding |
|---|---|---|
| C1 | **Critical** | The figure script replaces Hessian points that fall below DA with interpolated values, so failures are hidden and the plotted residuals are not raw data. |
| C2 | **Critical** | For nX=1e4 the curve labelled "Hessian" is actually DA data. The script also appends synthetic "dive" points, and the "fixed" curve has no legend entry (this is defect 2d). |
| C3 | **Critical** | The "Pure DA" JSON cannot be ground truth. It has R=0 for all L ≤ 100 km at nX=1e5–1e7, is within 1e-6 of the best known value at only 0–12 % of points, and the current script cannot have produced it. |
| C4 | **Critical** | No provenance: all `JAX_variants/`, all `outputs/`, and every `*.json` are untracked or gitignored. Nothing is seeded. |
| H1 | High | Each optimizer clips R to 0 and freezes the parameters, even though the physics returns an unclipped l/N with usable gradients beyond the horizon. |
| H2 | High | Objective scaling: the tolerances are absolute, while R runs from 1e-2 down to 1e-9. This causes premature stops near the horizon. |
| H3 | High | Constraints are handled by a 1e250 step penalty that has no gradient. The NumPy and JAX penalties differ. The FoR DA fallback scores infeasible points as better than feasible ones. |
| H4 | High | FoR stores the *regularized* objective as the key rate. Its Tikhonov polish biases the optimum by an amount comparable to R in the tail. |
| H5 | High | Defect 2b is real. H switches branch about 6 km too early, losing up to about 7 % of R over 149–153 km at nX=1e9. |
| H6 | High | `e_mis` is inconsistent: 0.01 in every optimizer, 5e-3 in `src/qkd/utils.py` and in the notebook header. The L=0 value does not match the 1.23e-3 you quoted. **Needs your decision.** |
| M1 | Medium | Defect 2c: the bounds are identical. The P_X "floor" is whatever value each method leaves behind after R hits 0. |
| M2 | Medium | Roughness (TV) is dominated by the region beyond the horizon: 97 % for DA at nX=1e4. |
| M3 | Medium | Horizon is defined as R ≥ 1e-12 (an invented "commercial insecurity limit"), not as l > 0. |
| M4 | Medium | Defect 2a cannot be reproduced from any JSON on disk. The positive-residual symptom is fully explained by C1. |
| M5 | Medium | Runs on 400-point and 1000-point grids are compared via `np.interp`. The CPU platform is hard-coded. Figures carry no git hash or config. |
| L1 | Low | Labels: "Trust-NCG" is actually `trust-constr`; the residual axis is labelled "Hz" but R is in bits/pulse; the FoR run is labelled "Hessian". |
| L2 | Low (physics, not changed) | Internal clamps can create flat regions, and `m_Z_total` sums n_X terms. This is flagged only, per the "do not change physics" constraint. |

---

## 1. Constraints, bounds, infeasible points and NaNs, per variant

All six variants use **byte-identical bounds**:
`[(4e-4,0.9),(2e-4,0.5),(1e-12,0.999)×3]`. See `BB84_Hessian_Optimization.py:229`, `BB84_PureDualAnnealing.py:223`,
`BB84_Hessian_FoR_Optimization.py:411`, `BB84_HighPerformance_LongDistance.py:231`,
`BB84_UltraPrecision.py:215` and `BB84_Parameters_2014_Optimization_Jax_updated.py:206`. The notebook uses `1-1e-12`
instead of 0.999 (`notebooks/BB84_Parameters_2014_Optimization_Jax_updated.py:194`).

| Concern | Where | What happens |
|---|---|---|
| Box bounds | SciPy `bounds=` (L-BFGS-B, trust-constr, DA) | Box only. `mu1 > mu2+mu3` and `P_mu1+P_mu2 < 1` are **not** in the box. |
| Coupled constraints | `src/qkd/model.py:114-138` | `jnp.where(cond, 0, 1e250)` subtracted from R. The step has **zero gradient**. At an infeasible point I measured obj = 2e250 with grad = ∇R only, so L-BFGS-B sees a gradient that does not point back to feasibility. `penalty_sum` (`:125`) is a no-op because P_mu3 is defined as 1−P_mu1−P_mu2. |
| NumPy penalty | `src/qkd/model_numpy.py:89-106` | Docstring says "Matches JAX implementation exactly". It does not: it uses 1e6·(1+v²) and a 0.999 cap, and omits the mu2/mu1 term. |
| mu2 lower bound | all scripts | `mu2 ≥ 2e-4 = mu3`, while the physics requires `mu2 > mu3` strictly. The box edge is infeasible. |
| Infeasible l ≤ 0 | H `:133-144`, FoR `:151-156`, Hyb `:133-136`, Ultra/updated `:117-120`, notebook `:287-290` | Reported R is set to 0 and the **warm start is frozen** at the last good point. H stores `current_best_guess`, not the optimizer output. |
| Non-finite value | H `:58`, DA `:75,84`, all wrappers | Returns `1e9` with **zero gradient**. That is a flat plateau, and L-BFGS-B reports "converged". |
| Gradient clipping | DA/Hyb/Ultra/updated `objective_wrapper` | `np.clip(grad, ±1e6)`. Harmless in practice because \|∇R\| ≤ 1e-2. |
| "R > 1 is garbage" | FoR `:147-149` | Sets rate=0. No point exceeds 1 in any stored run. |
| **FoR DA fallback** | FoR `:173-174, 191-192, 197-198` | Infeasible points, exceptions and R>1 all return cost **0.0**. Feasible points beyond the horizon have cost −R > 0, so DA *prefers* infeasible or garbage regions. |
| Exceptions | every variant | Records rate 0 and params `[0,0,0,0,0]` (or `start_guess` in FoR `:220`) with no status flag. |

### Is R clipped to 0 so that gradients vanish?

**The physics is not clipped.** `calculate_R = l/N` (`src/qkd/physics.py:289-295`). I measured at the
last positive point of H, nX=1e9 (L=189.5 km):

| L (km) | R | \|∇R\| |
|---|---|---|
| 189.5 | +3.8e-9 | 9.5e-12 |
| 191.5 | −8.0e-9 | 9.2e-8 |
| 200 | −4.3e-8 | 3.7e-7 |

Over 200 random feasible points at L=200 km, nX=1e4, **none** had a zero gradient and none were non-finite.
So the signal beyond the cliff exists. The **optimizer layer** throws it away: it clips the reported R, freezes
parameters, and the 1e9/0.0 fallbacks create artificial plateaus.

**Recommendation.**
- Optimize the unclipped `l/N`.
- Add a log-safe transform for conditioning, e.g. `−asinh(R/R_s)` with `R_s` set per (L, nX), or `−log(R)` once R > 0.
- Define the **horizon as the largest L with l > 0**, refined by root-finding in L at the optimum rather than read off the grid.
- Report l and R as-is (negative allowed) together with a status flag, and clip only in plots.

### Scaling (H2)

L-BFGS-B's `ftol` test is `(f_k−f_{k+1})/max(|f_k|,|f_{k+1}|,1)`. Because |f| = R ≪ 1, it acts as an **absolute** test.
- With `ftol=1e-12` (DA `:103`, Hyb, Ultra, updated, notebook `:281`), the run stops at about 1e-6 relative precision when R ≈ 1e-6, and at about 1e-4 near the horizon.
- `gtol=1e-12…1e-14` is also absolute. \|∇R\| ≈ 1e-11 at the horizon (see above), so some points stop on iteration 0.

Your target of being within 1e-6 relative of the reference is therefore unreachable with the current tolerances in the tail.

## 2. Known defects

### 2a. Glitches at about 23, 30 and 57 km (nX=1e9): P_X→0, mu2/P_mu2 at bounds, yet positive residual

- **Cannot be reproduced from data on disk.** I scanned every grouped JSON in the repo, excluding NN training data: all
  four H runs, both FoR runs, and the Baseline, Hybrid, Ultra, notebook and tier3 outputs. None has a point with
  15 < L < 65 km, R > 0 and P_X < 0.05 or mu2/P_mu2 at a bound. There are also no isolated parameter spikes there in any run.
  The glitch most likely came from a run whose outputs have since been deleted. `git status` shows many deleted
  `Optimization/outputs/**` files, and all `outputs/JAX` data is untracked (C4), so it cannot be recovered.
- **Why the residual looked positive: a plotting bug, not physics.**
  `outputs/JAX/generate_paper_plots.py:159-170` marks every Hessian point with `R_H < 0.99·R_DA` (or R_H = 0) as
  "invalid" and **overwrites it with a linear interpolation of its neighbours**, provided fewer than 15 % of points are
  affected. The residual panel (`:176, :224`) is computed on the overwritten array. A glitch point where the Hessian lost
  to DA is therefore replaced by its neighbours' value, which is ≥ 0.99·DA and usually > DA, so the residual comes out positive.
- **Mechanism for P_X → 0 with params at bounds.** The Hessian path is consistent with the two H1 failure modes:
  - Non-finite values become `1e9` with a zero gradient (`BB84_Hessian_Optimization.py:58`).
  - `trust-constr` with a zero-gradient step penalty can be pushed onto the box faces.
  Neither failure is flagged, and the comparison script hides it.
- Verdict: **a bug**, in the optimizer's failure handling plus the plot "cleaning". It is neither a segment boundary nor a warm-start issue.
  *Action needed from you:* if you still have the figure or log, tell me its filename.

### 2b. Discontinuous jump near 148 km (P_mu1 0.24→0.40) vs baseline at about 155 km

Confirmed. These are two competing local optima, and the H continuation **leaves branch A too early**.

| run (nX=1e9) | P_mu1 jump |
|---|---|
| H | 148.4 km, 0.239 → 0.396 |
| Base, Ultra | 154.2 km, 0.247 → 0.427 |
| FoR | none (Tikhonov keeps it on one branch; see H4) |
| DA | 8 jumps between 137 and 176 km (noise) |

I froze H's parameters at 140 km (branch A) and at 160 km (branch B) and evaluated both across L:

| L | R_A (frozen@140) | R_B (frozen@160) | R_H stored | R_Base stored |
|---|---|---|---|---|
| 148 | 6.783e-6 | 6.168e-6 | 6.858e-6 | 6.819e-6 |
| 149 | **6.300e-6** | 5.800e-6 | **5.883e-6** | 6.336e-6 |
| 152 | 4.985e-6 | 4.797e-6 | 4.854e-6 | 5.027e-6 |
| 154 | 4.213e-6 | 4.206e-6 | 4.249e-6 | 4.260e-6 |
| 155 | 3.855e-6 | 3.931e-6 | 3.969e-6 | 3.951e-6 |

- The branches cross at about 154–155 km.
- From 149 to 153 km, H's stored R is lower than even branch A with *frozen 140 km parameters*. That is a regret of about 7 % at 149 km.
- Before 148 km, H beats Base by about 0.6 %, so Base is not optimal either.
- Under C1's cleaning these points have ratio < 0.99 and would be interpolated away in the paper figure.

This directly motivates the branch check planned for Phase 2.

### 2c. P_X floor about 0.1 (Hessian) vs about 0 (DA)

**The bounds are identical.** The difference exists only **beyond the horizon** (R=0), where the parameters have no meaning:

| run | nX | P_X just before horizon | P_X beyond horizon |
|---|---|---|---|
| FoR | 1e4 | 0.110, 0.100, 0.061 | constant **0.100** |
| H | 1e9 | 0.33, 0.30, 0.25 | constant 0.251 (frozen `current_best_guess`) |
| DA | 1e4 / 1e9 | … 0.065 / … 0.061 | 4e-7 … 3e-4, wandering over 105 / 20 distinct values |

R is identically 0 there and DA has nothing to optimize, so it returns random states near the P_X lower bound.
The Hessian variants return frozen warm starts.

Fix: report parameters beyond the horizon as NaN, with status `beyond_horizon`.

### 2d. "DA" curve dies at about 171 km in the Fixed vs Optimized plot, while pure DA reaches about 190 km

That curve is **not DA. It is the "fixed" curve**:
- `generate_paper_plots.py:405-418` builds it by taking the optimized parameters at **L=0** from the
  plotted run (FoR, or DA for nX=1e4 because of the patch at `:366-371`) and holding them constant for all L.
- It is drawn at `:438` as a **dashed line with no label**, in the same colour as the dashed DA curve (`:432`), whose legend entry is "DA".

Recomputed:

| nX | fixed parameters | fixed horizon (R ≥ 1e-12, and also l > 0) | optimized horizon |
|---|---|---|---|
| 1e4 | `[0.613, 0.138, 0.046, 0.611, 0.415]` | **127.3 km** | 147.4 |
| 1e9 | `[0.490, 0.368, 0.186, 0.773, 0.877]` | **169.9 km** | 189.5 |

These match `outputs/JAX/Comparison/qkd_horizon_comparison.md` exactly.

Fix:
- Label it "Fixed (params optimal at L=0)" and give it a distinct style.
- Draw the DA curve separately.
- State the definition in the caption.

Related problems in the same figure:
- **`:366-371`** silently replaces the nX=1e4 "Hessian" data with DA data, while keeping the Hessian label.
- **`:380-385, :396-398`** append a fabricated point (L_last+1 km, R=1e-15) to "force the dive".

Both must go. Neither change is disclosed in the figure.

## 3. Roughness metric

The current metric (`generate_paper_plots.py:89-98, :276-280`) is the total variation of all 5 parameters over **all** L,
including points with R=0 and points left by failed optimizations.

| run | nX | TV over all L | TV where R>0 | share beyond horizon |
|---|---|---|---|---|
| DA | 1e4 | 90.3 | 3.06 | **97 %** |
| DA | 1e9 | 24.8 | 6.29 | 75 % |
| H | 1e4 | 1.77 | 1.54 | 13 % |
| FoR | 1e9 | 3.06 | 3.06 | 0 % |

So the published "DA is rough" result is mostly measuring noise in a region where nothing is being optimized.

**Proposed definition.**
- Let S = {i : R_ref(L_i) > τ·R_ref(0)} with τ = 1e-3, so the region is set by the *reference*, not by the method being scored.
- For each parameter, normalize to its bound range, Δp̃ = Δp/(ub−lb), so that mu and P are comparable.
- Report two numbers per method and nX:
  - **TV_S** = Σ over consecutive pairs both in S of ‖Δp̃‖₁, divided by the L-span of S, in units of 1/km.
  - **Jumps** = the count of ‖Δp̃‖∞ > 0.05 in S, listing each jump's L. This separates genuine branch switches like the one at ~154 km from jitter.
- Use the same L grid for all methods, so no interpolation is needed.

## 4. Duplicates, stale scripts and results; consolidation plan

Inventory:
- **Untracked (C4):**
  - `JAX_variants/` (6 scripts)
  - all of `Optimization/run_scripts/outputs/` (271 MB), including the DA reference JSONs
  - `tools/compare_optimization_runs.py`
  - `notebooks/*.py`
  - The `.gitignore` rule `*.json` ignores every result file. The negation line also carries an inline comment
    (`!**/Training_Data/**/*.json  # Keep…`), which gitignore does not support, so that line does not work as intended.
- **Tracked but deleted or moved in the working tree:** `Optimization/JAX_notebook.py`,
  `Optimization/run_scripts/{legacy_global,legacy_notebook_logic,plot_results,tier1_*,tier2_local,tier3_*}.py`
  (copies now live under `run_scripts/outputs/**`), and `Optimization/outputs/**`. The tree on `main` is dirty.
- **Identical copies:**
  - `Optimization/NOTJAX.py` ≡ `notebooks/BB84_Parameters_2014_Optimization_Jax_old.py` (same md5).
  - `Optimization/BB84_Parameters_2014_Optimization_Jax_updated.ipynb` and `notebooks/…_updated.ipynb` share a name but differ.
  - About 10 duplicate JSON pairs, e.g. `Optimization/dataset/qkd_grouped_dataset_20260120_031938.json` ≡
    `outputs/legacy_notebook_logic/reordered_…_031938.json`, and four identical `tier3_global_jax/dataset/reordered_*` files.
- **Near-duplicate scripts:** the 6 variants share about 80 % of their code (setup, I/O, plotting, constants).
  `BB84_Parameters_2014_Optimization_Jax_updated.py` duplicates the notebook loop.
- **Stale or broken:**
  - `BB84_PureDualAnnealing.py:89-143` never appends results on the success path, and `current_best_guess` is never updated. It cannot have produced the stored Pure_DA JSON, which has no `run_metadata.json`.
  - Four H runs exist, and the latest three have identical spikes (nX=1e8, 158 km).
  - The `Comparison/` folder holds about 50 timestamped PNGs from successive script versions, with no record of which code produced each.
  - Repo-root `tier*.log`, `temp_generate_plot.py`, `time_test.py` and the root `qkd_*_20260205_115429.json`.

**Plan.** These are small commits on a new branch. Nothing is deleted; old material moves to `archive/`.
1. Commit the current scripts **as-is** first, so they get provenance (needs your OK, see the decisions list).
2. `git mv` into `Optimization/archive/`:
   - `JAX_variants/*`
   - `run_scripts/outputs/*.py` and the tier scripts
   - `NOTJAX.py`
   - the old notebook `.py` exports
   - `outputs/JAX/generate_paper_plots.py`, with a header noting C1/C2
3. Freeze the DA data under `Optimization/reference/legacy_da/`, with a `PROVENANCE.md` that records the defects in C3.
   Its role is "frozen legacy comparison only", not ground truth.
4. New layout:
   - `src/qkd_opt/` holds the optimizer (`reparam.py`, `objective.py`, `optimize.py`).
   - `tests/` holds the tests.
   - `Optimization/benchmark/run_benchmark.py` is the single entry point.
   - `Optimization/reference/` holds the dense-reference generator and its output (JSON plus a SHA-256 manifest).
5. Results are written to `results/<git-sha>_<config-hash>/`. `.gitignore` keeps ignoring bulk JSON, but the
   reference and benchmark-summary JSONs are explicitly un-ignored.

## C3 detail: the "Pure DA" reference

Regret against the best-of-{DA, H, FoR, Hyb} envelope on the shared 400-point grid, restricted to points where the envelope R > 0.
"ok" = the share of points within 1e-6 relative of the envelope.

| nX | DA median / ok | H median / ok | FoR median / ok | Hyb median / ok |
|---|---|---|---|---|
| 1e4 | 2.6e-3 / 0 % | 0 / 97 % | 4.9e-9 / 74 % | 1.7e-11 / 95 % |
| 1e5 | **1.0** / 0 % | 0 / 98 % | 1.3e-9 / 75 % | 9.3e-13 / 88 % |
| 1e6 | **1.0** / 0 % | 0 / 95 % | 5.6e-10 / 75 % | 5.6e-12 / 85 % |
| 1e7 | **1.0** / 0 % | 0 / 98 % | 3.6e-10 / 77 % | 1.7e-12 / 81 % |
| 1e8 | 4.0e-4 / 10 % | 1.6e-13 / 95 % | 3.0e-10 / 80 % | 3.6e-13 / 80 % |
| 1e9 | 1.4e-4 / 12 % | 0 / 95 % | 1.4e-9 / 77 % | 4.1e-12 / 81 % |

What this shows:
- At nX = 1e5–1e7, **DA has R=0 for every L ≤ 100 km**. In the script, L ≤ 100 is not DA at all: it is a cold-started
  L-BFGS-B from the 1e9 default guess, because only nX=1e4 and 1e9 have entries in `initial_guesses_map`, and the warm start is never updated.
- DA runs are unseeded, and `no_local_search=True` gives at best about 1e-4 relative precision.
- The envelope above is *not* a reference either. Phase 1's dense Sobol plus refinement is required before regret numbers mean anything.

## H4 detail: FoR stores the regularized objective

`BB84_Hessian_FoR_Optimization.py`:
- Lines 60-73 add `λ‖p−p_prev‖²` to the objective. λ is 1e-6 in the sweeps and **1e-4** in the "manifold polish" (`:296`).
- Line 320 stores `-res_polish.fun`, so the stored "key rate" includes the penalty.
- Verified: at L=0, nX=1e9, FoR stores 1.08996e-2 while R(params) = 1.08997e-2.
- In the tail, λ·‖Δp‖² ≈ 1e-4·(1e-2)² = 1e-8, compared with R ≈ 1e-7 to 1e-9. The bias is comparable to the signal, so FoR's smoothness is partly bought with regret.

## H6: e_mis and the L=0 reference value

- Every optimizer and plot script uses `e_mis = 0.01`. `src/qkd/utils.py:25` and the notebook header (`:109,113`) use `5e-3`, which is the value labelled "given in the paper".
- R(L=0, nX=1e9) at the stored optimum is **1.090e-2** with e_mis=0.01 and **1.285e-2** with e_mis=5e-3. All runs agree on 1.0900e-2.
- Your expected values (Lim et al. ≈ 1.25e-3, ours 1.23e-3) are about 10× lower. Either they refer to a different quantity or setting
  (a different L, P_X definition or units), or there is a discrepancy that should be resolved before the regression test is written.

## M5 / L1 / L2: smaller items

- **Different grids:** `linspace(0,200,400)` for H, FoR, DA and Hyb vs `linspace(0,200,1000)` for Base and Ultra. Comparisons interpolate R linearly across a log-scale cliff.
- **Platform and seeding:** `jax_platform_name="cpu"` is hard-coded in every variant, which blocks the A100 target. `np.random` (Hyb/Ultra/updated/notebook) and `dual_annealing` are unseeded.
- **Processes:** `ProcessPoolExecutor` combined with JAX works on macOS (spawn), but on Linux the default fork would follow JAX's thread-pool initialization. Not a concern for the new vmap design.
- **Labels:** the legend says "Hessian (Trust-NCG)", but the code uses `method='trust-constr'`, and the plotted data is FoR (`HESSIAN_STRATEGY="Hessian_FoR"`). The residual axis says "Hz", but R is in bits/pulse.
- **Physics (not changed, flagged only):**
  - `model.py:84` sets `m_Z_total` to the sum of the **n_X** terms; the original author left a comment questioning this.
  - Clamps such as `max(·,0)` in S_0 and S_1, Phi clipped to [−10, 10] then min 0.5, and gamma capped at 10 can create zero-gradient sub-regions. My random probe found none at L=200 km, but the new optimizer must detect them, not assume they never occur.

## Tooling available (relevant to Phase 1/2)

The `.venv` has jax 0.9.0, scipy 1.17.0 and numpy 2.4.2. **optax, optimistix and pytest are not installed.**
`scipy.stats.qmc.Sobol` is available, so no new dependency is needed for Sobol sampling.
