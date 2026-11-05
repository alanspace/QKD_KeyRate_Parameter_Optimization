# CLAIMS.md: audit of numeric claims

Sources audited:
- `result_report/results_report.tex` (PDF: `results_report.pdf`)
- `assets/Optimization_Speedup_Math.tex` (PDF: `Optimization_Speedup_Math.pdf`)
- `ReadMe.md`

**No report number has been edited by hand.** This file only classifies the existing numbers and points
to the script that regenerates a correct replacement, where one exists.

**Status values.**
- **reproduced**: a script in this repo regenerates the number under the stated definition.
- **not reproduced**: either no script or stored measurement produces it, or the regenerated value disagrees.
- **ill-defined**: the quantity as defined compares unequal work, mixes units or regimes, is modelled
  instead of measured, or is computed in a way that cannot support the claim. Re-running would not make it
  meaningful.

Line numbers refer to the working-tree files on 2026-11-05. `results_report.tex` has uncommitted edits
that predate this work.

**Regenerating scripts.** Every replacement value quoted below is copied from one of these outputs; each
output is stamped with git commit, config hash and hardware (Apple M2 Pro, CPU, JAX 0.9.0, float64).

| ID | Script | Output |
|---|---|---|
| **[B-leg]** | `Optimization/benchmark/run_benchmark.py --preset legacy` | `results/benchmark_legacy_501fa1d36e_7e973d7467b6/RESULTS.md` |
| **[B-lim]** | `Optimization/benchmark/run_benchmark.py --preset lim_exact` | `results/benchmark_lim_exact_76155eb48a_128f57edfbb5/RESULTS.md` |
| **[T]** | `Optimization/benchmark/timing_benchmark.py` | `results/timing_76155eb48a_b1cac5b411b3/TIMING.md` |
| **[A-leg]** | `Optimization/analysis/analyses.py --preset legacy` | `results/analyses_legacy_af299a0fb7_a829421e2844/ANALYSES.md` |
| **[A-lim]** | `Optimization/analysis/analyses.py --preset lim_exact` | `results/analyses_lim_exact_af299a0fb7_dcc8a16d9eb7/ANALYSES.md` |
| **[P]** | `Optimization/analysis/physics_diff.py` | `results/physics_diff_76155eb48a_dba94a5ecdc3/PHYSICS_DIFF_TABLES.md` |

**Blocked: NN surrogate claims.** No claim about the neural network can be reproduced in this
environment, for three reasons:
- `.venv` has no torch, scikit-learn or joblib.
- `NeuralNetwork/models/bb84_nn_model.pth` has 6 outputs, but every evaluation script builds a 5-output
  net. The scaler and model pairings are not recorded anywhere.
- All evaluation scripts feed e_mis = 5e-3 to a model trained on e_mis = 0.01.

Installing torch and scikit-learn needs your approval; see the decisions list in the hand-off.

---

## 1. `result_report/results_report.tex`

| # | line | claim (verbatim, abridged) | status | why / regenerated replacement |
|---|---|---|---|---|
| R1 | 22 | ">5,900,000× raw speedup … (Legacy 329 ms vs NN 0.056 µs)" | **ill-defined** | Compares one full optimisation against a batch-1000-amortised NN forward pass. The 329 ms is typed into `verify_report_metrics.py:142-144`. NN not runnable (blocked). Measured on this machine **[T]**: legacy DA (archived call) takes 3.3–9.9 s per point at L > 100 km; the warm-started JAX solver takes 297 µs median batch-1 device-resident (578 µs end-to-end). |
| R2 | 22 | "~11,000× system speedup … (1.2 µs)" | **ill-defined** | 11,000× = 329 ms / 28 µs, while the same sentence says 1.2 µs. No `benchmark_inference.py` exists. |
| R3 | 22, 179, 284, 287 | "129× improvement in parameter stability" (Roughness 0.7 vs 90.3) | **ill-defined** | The TV is summed over all L including the R = 0 region, where 97 % of DA's TV lies (REVIEW.md §3). The row is labelled n_X = 1e9 but the data are n_X = 1e4. 129 is a ratio of rounded numbers. FoR is Tikhonov-regularised and its stored rate includes the penalty. Replacement **[B-leg]**: TV/km of box-normalised parameters on S = {R_ref > 1e-3·R_ref(0)}. At n_X = 1e4: new_warm 1.8460e-03, legacy_DA 7.9460e-03 (6 jumps), legacy_Hessian 1.8461e-03, legacy_Hessian_FoR 1.8430e-03. At n_X = 1e9: 1.8616e-03 / 2.6100e-03 / 1.8388e-03 / 1.8468e-03. |
| R4 | 22, 234 | "20 km+ security reach boost … ~16 % efficiency leap" | **ill-defined** | "Fixed" means parameters optimal at L = 0 held constant. The horizon uses an invented R ≥ 1e-12 threshold. "Optimized" is the FoR optimizer, not the NN. ~16 % holds only for n_X = 1e4 and 1e5. No regenerated equivalent yet; optimized horizons with l > 0 (legacy) are in **[B-leg]**. |
| R5 | 47 | "6,000 ground-truth optimal points … Multi-Start … guarantees global maximum" | **not reproduced** | The current model trained on 2,117 rows (FoR run, 6 × 400 grid, R > 0). FoR has no multi-start, and "guarantees" is unsupported. **[B-leg]** shows legacy_Hessian_FoR is within 1e-6 of the reference at only 80.9 % of points (max regret 96.47). |
| R6 | 52-56 | architecture: 4 features, 16-32-16, 5 outputs, 5,000 epochs | **not reproduced** | The current checkpoint has 6 outputs. Two of the four features are constant in the training data. |
| R7 | 69 | "probability terms (< 1e-10)" | **ill-defined** | Qualitative, no source. |
| R8 | 74 | "near 0 % error … (0-150 km)" | **not reproduced** | Blocked (NN). The original evaluation used training points and an off-distribution e_3 input. |
| R9 | 79 | "20 % hardware drift … P_X near 100 % collapse … intensities < 5 %" | **ill-defined** | Linear ±20 % perturbation clamped to bounds, unseeded DA, floored at 0. Replacement **[A-leg]** / **[A-lim]**: logit-space, both signs, regret with others frozen and with others re-optimised, at 5 (L, n_X) points. |
| R10 | 93 | "S_μ ≈ 0 (< 50 km) … S_μ → ∞ at 180 km" | **not reproduced** | No script. Gradients are finite at 180 km (REVIEW.md §1). |
| R11 | 105 | "709 s → 78.9 s (9× Algorithmic Speedup)" | **ill-defined** | Ratio of "average per n_X curve" times. Six curves ran concurrently, compile time is included, and the DA baseline was L-BFGS-B below 100 km and gave R = 0 at all L ≤ 100 km for n_X = 1e5–1e7. Wall times were 714 s vs 308 s. Replacement, measured **[T]** and **[B-leg]**: legacy DA 3.3–9.9 s/point (L > 100 km, archived call, single process). new_warm sweep 2.1 s steady for 2,400 points (0.9 ms/point) plus 38.2 s JIT compile. |
| R12 | 106 | "XLA compilation (> 460×) … total 3,574×" | **ill-defined** | Modelled, not measured (see S4–S12). |
| R13 | 113 | "709 s → 1.2 µs (5.9M× Speedup)" | **ill-defined** | Divides a per-curve time by a per-point time. 709 s / 1.2 µs = 5.9 × 10⁸, not 5.9 × 10⁶. |
| R14 | 121 | "1.2 µs inference time" | **not reproduced** | Typed into `generate_comparison_table.py:16`; no measurement exists. |
| R15 | 135 | "ε ≈ 1e-3 … 1e-16 … 1e-4" | **ill-defined** | No source. Measured precision of each method vs reference is in **[B-leg]**. |
| R16 | 138, 147 | "5,000,000× throughput gain" | **ill-defined** | Older version of R1. |
| R17 | 144 | "N_iter ≈ 1000+" | **not reproduced** | No source. **[T]** logs DA nfev of 50,005–150,013 per point. |
| R18 | 150, 300 | "> 150,000× total speedup"; "conservative > 6,000×" | **ill-defined** | Built from typed constants (0.329 s, 12 ms, 2 µs). |
| R19 | 158 | "Approx (< 0.01 % error)" | **not reproduced** | No metric computes it. |
| R20 | 159 | "Setup Time 12 hours / 78.9 s / 6 mins" | **not reproduced** | 12 h and 6 min have no source. 78.9 s is a per-curve average, not a setup total. |
| R21 | 160 | "Inference ~329 ms / ~12 ms / ~0.056 µs" | **ill-defined** | Typed constants. "Inference" means an optimisation for two columns and a batched forward pass for the third. |
| R22 | 161 | "System Latency (Batch 1) ~1.2 µs" | **not reproduced** | Contradicts the 28 µs at l. 176/208. |
| R23-24 | 162-163 | "3,574× / > 5,900,000×", "27× / ~11,000×" | **ill-defined** | Mixes modelled per-curve and per-point ratios. |
| R25 | 174 | "consistent with the ground truth (< 1e-10 diff)" | **not reproduced** | The script only checks R > 0 at one point, and with current files it SKIPs. |
| R26 | 176, 191, 208 | "~28 µs" batch-1 latency | **not reproduced** | Timed without `torch.no_grad()`; a hard-coded fallback prints 28 µs. Blocked. |
| R27 | 182, 191, 213 | "drifted +5 % … retained 78.4 % of the optimal Key Rate" | **ill-defined** | The ratio R(p_nom, α=0.21)/R(p_nom, α=0.2) is mostly the extra 1 dB channel loss, not robustness. **[A-leg]** at L = 100 km, n_X = 1e9, α′ = 0.21 with optimiser parameters: old ratio **0.783**, true regret vs re-optimised **1.15e-05**. The NN is blocked. |
| R28 | 185, 214, 224 | "evaluated at 200 km … near-zero"; "Training Boundary (180 km)" | **ill-defined** | Vacuous test (R_opt(200 km) ≤ 0, so it cannot fail). Training data reach 189.5 km. Replacement extrapolation tests (held-out n_X, interior L, P_dc / e_mis transfer) of a lookup-table surrogate are in **[A-leg]** / **[A-lim]**; the NN is blocked. |
| R29 | 188 | "Fiber Length and Misalignment Error are the dominant drivers" | **ill-defined** | Unscaled input, single point, and e_mis is constant in training data. |
| R30 | 191 | "deterministic seeds … numerically reproducible" | **not reproduced** | DA, `random_split` and `randn` are unseeded. (The new pipeline is seeded.) |
| R31 | 197 | "Bell Curve centred at 0 % with no outliers > 5 %" | **ill-defined** | Absolute error cannot be centred at 0, `xlim(0, 5)` hides outliers, and it uses an old dataset. |
| R32 | 207 | "5,247,143×" | **ill-defined** | Same construction as R1. |
| R33 | 209, 300 | "Smoother Control 31 % / 99.9 %" | **ill-defined** | 99.9 % compares against synthetic noise; 31 % comes from an old notebook table. See R3 for the replacement. |
| R34 | 210 | "Relative Error < 1 % — Verified" | **ill-defined** | The script only checks that PNG files exist. |
| R35 | 211 | "Gain at 180 km 52.0× / 52.2×" | **ill-defined** | Circular: computed from the document's own typed numbers. |
| R36 | 212 | "Manifold Consistency — Valid" | **ill-defined** | A single point, and it SKIPs with the current files. |
| R37 | 241-249 | **Table 4** "Key Rate Gains" (4.2e-4 / 4.4e-4 / 1.05×; 1.8e-5 / 2.5e-5 / 1.38×; 1.8e-9 / 9.4e-8 / ~52×) | **not reproduced** | n_X, e_mis and the static parameter set are not stated, and the "AI" values match no stored run at any single n_X. Optimised R at the reference points is in `Optimization/reference/reference_legacy.json`. |
| R38 | 253, 258 | "n_X = 1e5 regime"; "drops significantly for n_X < 1e7" | **not reproduced** | The source data directory is empty. Horizons per n_X are in **[B-leg]** / **[B-lim]**. |
| R39 | 284 | "+6.57e-10 Avg Rate Improv" | **ill-defined** | A stale old-notebook number placed next to TVs from a different run and n_X. |
| R40 | 300 | "329 ms / 12 ms / 0.002 ms per point"; "< 1 % … 5 % drift" | **ill-defined** | See R18, R21, R27, R34. |

## 2. `assets/Optimization_Speedup_Math.tex`

This file is untracked. The whole document is a hand calculation, and no script produces it.

| # | line | claim | status | why |
|---|---|---|---|---|
| S1 | 10 | "The 3,574× Hybrid Speedup" | **ill-defined** | Modelled numerator (282,000 s) over a measured per-curve average (78.9 s). |
| S2 | 21-23 | ΔL = 0.5 km, N_p = 400 | **reproduced** | `linspace(0, 200, 400)`; the actual ΔL is 0.5013 km. |
| S3 | 27-28 | "N_e ≈ 15,000 evaluations/point" | **ill-defined** | That is the DA *maxiter* for L > 160 km only, not nfev. **[T]** measured nfev = 150,013 at L = 190 km, n_X = 1e9. |
| S4 | 34-36 | "t_Py ≈ 47 ms" | **not reproduced** | No measurement. **[T]**: 66–67 µs per jitted-JAX objective evaluation inside DA. |
| S5-S6 | 41-42 | "705 s/point", "282,000 s = 78.3 h" | **ill-defined** | Modelled. The per-point model coincides with the measured per-*curve* 705–713 s. |
| S7 | 50, 55 | "t_JAX ≈ 102 µs", "1.53 s/point" | **not reproduced** | No measurement. |
| S8 | 58 | "T_total,JAX = 709.0 s (empirically measured)" | **ill-defined** | A per-curve average of 6 concurrent workers, compile included (see R11). |
| S9 | 62-64 | "460×", "> 100× on slides" | **ill-defined** | Modelled. |
| S10-S11 | 69-73 | "78.9 s", "8.99×" | **ill-defined** | See R11. |
| S12 | 81 | "3,574×" | **ill-defined** | Internally inconsistent: 460 × 8.99 ≈ 4,135. |
| S13 | 87 | "78.3 hours → 78.9 seconds" | **ill-defined** | Modelled vs measured. |

## 3. `ReadMe.md`

| # | line | claim | status | why / replacement |
|---|---|---|---|---|
| M1 | 19, 25 | "> 6,000-fold practical speedup over traditional legacy solvers" | **ill-defined** | 6,000 = 12 ms / 2 µs (JAX L-BFGS-B vs NN), not legacy. The NN is blocked. Measured alternatives **[T]**: legacy DA 3.3–9.9 s/point (L > 100 km); JAX warm solve 297 µs (device) / 578 µs (end-to-end); lookup table 8.2 µs / 10.7 µs. |
| M2 | 23 | "~47x - 60x higher key rate at long distances (180 km+)" | **not reproduced** | No producer, and the static parameter set is unspecified. |
| M3 | 24 | "+5.0 km extended secure communication distance" | **not reproduced** | No stored output. It contradicts the report's "20 km+". |
| M4 | 108 | "Legacy (Dual Annealing) > 10.0 s" | **not reproduced** | **[T]** measured 3.3–9.9 s/point for the archived DA call at L > 100 km, and 1–2 ms for its L-BFGS-B branch at L ≤ 100 km. |
| M5 | 109 | "Modern (JAX/SciPy) 0.01-0.05 s, 1,000×" | **ill-defined** | 10 s / 0.01–0.05 s is 200–1,000×. No measurement is stored. |
| M6 | 110 | "Neural Network 0.000002 s, > 6,000× - 140,000×" | **not reproduced** | Blocked. 140,000 is unexplained. |
| M7 | 117 | "Runtime (6000 pts) ~1 hr 30 min → ~2 min, ~45x" | **not reproduced** | Hard-coded `90/2` in `verify_report_metrics.py:145`; no matching log. |
| M8 | 124 | "< 5 % relative error in useful ranges" | **not reproduced** | No metric computes it. |

## 4. Other locations repeating these claims (not edited)

These files repeat claims from the tables above and were not changed:
- `streamlit_app.py:314-316, 382`
- `project_skills_profile.md:23`
- `feedback_on_the_project.md`
- `implementation_plan*.md`
- `rigorous_proof_and_extrapolation.md`
- `NeuralNetwork/Training_Data/dataset_summary.md`

The full inventory, with file:line producers for every number, was generated for this audit at
`/tmp/claims_inventory/inventory.md`. It is not committed; regenerate it on request.
