# src/qkd vs Lim et al. 2014: differences and paper targets

Reference: C. C. W. Lim, M. Curty, N. Walenta, F. Xu, H. Zbinden, *Concise security bounds for practical
decoy-state quantum key distribution*, PRA 89, 022307 (2014), arXiv:1311.7129.

I read the equations from the arXiv LaTeX source, `SecurityBounds_DecoyState_arXiv.tex`, specifically
Eqs. (1)–(5) and the "Evaluation" section. `src/qkd` line numbers refer to the **working tree**. It
contains uncommitted edits that predate this work; every result is stamped with `src_uncommitted_changes`
and a physics source hash.

All numbers in this file are copied verbatim from
`results/physics_diff_<commit>_<cfg>/PHYSICS_DIFF_TABLES.md`, which `Optimization/analysis/physics_diff.py`
generates. None are typed by hand.

## Presets

| preset | what it is |
|---|---|
| `lim_exact` | `src/qkd_opt/physics_lim.py`, transcribed literally from the paper (switches in `src/qkd_opt/presets.py`). Constants: e_mis = 5e-3, p_ap = 4e-2, p_dc = 6e-7, η_Bob = 0.1, α = 0.2 dB/km, f_EC = 1.16, ε_cor = 1e-15, μ3 = 2e-4, ε_sec = κℓ with κ = 1e-15 (greatest fixed point). R_k = D_k(1+p_ap), and e_k uses **η_ch** exactly as printed. |
| `legacy` | `src/qkd/model.py`, called unmodified. e_mis = 0.01, p_ap = 0, ε_sec = 1e-10 fixed. |

Validation: `physics_lim.py` run with the legacy switches reproduces `src/qkd` to better than 1e-8 relative
in ℓ at 40+ random interior points (`tests/test_qkd_opt_physics.py::test_matches_src_qkd_with_legacy_flags`).
Both presets pass second-order `jax.test_util.check_grads` at interior points.

## Differences between src/qkd and the paper

| # | item | paper | src/qkd | effect |
|---|---|---|---|---|
| (i) | Transmittance in the e_mis term of e_k | e_k = p_dc + e_mis[1 − exp(−**η_ch** k)] + p_ap D_k/2 (Evaluation section, verbatim) | `physics.py:204` uses the argument `eta_sys_values`, passed at `model.py:33`, i.e. **η_sys** = η_Bob·η_ch | With η_ch, the misalignment error does not scale with η_Bob, so QBER → e_mis/η_Bob = 5 % at long distance. With η_sys, QBER → e_mis. This is the largest single effect (chain stage 7). It may be a typo in the paper; see the targets below. |
| (ii) | m_Z in Eq. (4) | m_Z = Σ_k m_{Z,k}, the total Z-basis **errors** | `model.py:84`: `m_Z_total = jnp.sum([n_X_mu_1, n_X_mu_2, n_X_mu_3])`, the total X-basis **detections**. The original author's comment at `model.py:81-83` already doubts it. | √(m_Z/2·ln(21/ε)) is hugely overestimated, which inflates v_{Z,1} and φ_X. **A bug.** Chain stage 1. |
| (iii) | 6 log₂(21/ε_sec) term in Eq. (1) | −6 log₂(21/ε_sec) − log₂(2/ε_cor) | `physics.py:279` matches the paper. **But** `physics.py:133` clips ε_sec ≥ 1e-10 inside the √ term only. | Consistent while ε_sec = 1e-10 is fixed. It would be inconsistent if ε_sec = κℓ < 1e-10 were ever passed to src/qkd. |
| (iv) | ε_sec | ε_sec = κℓ, κ = 1e-15 | ε_sec = 1e-10, fixed | Chain stage 5. ε_sec = κℓ has the trivial fixed point ε = κ (ℓ ≤ 1); the greatest fixed point is used. Near the horizon the non-trivial root disappears, so R(p) is **discontinuous** there. |
| (v) | Detection rate with afterpulsing | R_k = D_k(1+p_ap), used for counts and N | N and the counts use D_k (`physics.py:69-71, 99-105`). p_ap enters only e_k. | Chain stage 4. With p_ap = 0 (legacy) it has no effect. |
| (vi) | p_ap, e_mis values | p_ap = 4e-2, e_mis = 5e-3 | 0 and 0.01 in all optimizer runs | Chain stages 2 and 3. |
| (vii) | n^±, m^± | n^-_k = e^k/p_k [n_k − δ], no clamp | `physics.py:143, 191` clamp n^-, m^- at 0 | Clamping at 0 still gives a valid (tighter) lower bound, so the literal formulas are more conservative. Chain stage 6: it lowers R where n_k − δ < 0, e.g. by about 21 % at L = 0, n_X = 1e4. |
| (viii) | s_{X,0}, s_{X,1} | lower bounds as printed | `physics.py:165, 175` clamp at 0. S_0 adds 1e-15 to its denominator. | Valid (true counts are ≥ 0). `physics_lim.py` clamps s_0 the same way and drops the single-photon term when s_1 ≤ 0. |
| (ix) | γ and φ_X | γ(a,b,c,d) as in Eq. (5). φ ≤ v/s + γ. | `physics.py:228-273`: b is sanitised (b ∉ (0,1) → 0.5), γ is capped at 10, v/s is clipped to [−10, 10], φ is capped at 0.5. γ receives v/(s+1e-12) (`model.py:102`) while Φ uses v/(s+1e-15). | Inactive in the interior; they create flat regions near degenerate points. `physics_lim.py` caps φ at ½ only (h is symmetric). |
| (x) | Floor in Eq. (1) | ℓ = ⌊…⌋ | no floor | A continuous relaxation, needed for optimisation. Both presets omit the floor. |
| (xi) | Expected-count model | The paper does not print it. The standard reading is n_{X,k} = N q_x² p_k R_k with N fixed by Σ_k n_{X,k} = n_X, and m_{X,k} = N q_x² p_k e_k (likewise for Z with (1−q_x)²). | `physics.py:58-127` is equivalent with R_k → D_k | No difference beyond (v). |
| (xii) | √ terms | δ = √(n/2 · ln(21/ε)), natural log | `physics.py:135`, natural log | Match. |
| (xiii) | λ_EC | n_X f_EC h(e_obs), e_obs = m_X/n_X | `physics.py:206-226`; e_obs capped at 0.5 | Match. |

## Quantified effect of each difference, and the paper targets

_(The tables below are pasted verbatim from the generated `PHYSICS_DIFF_TABLES.md`; see the results directory named there.)_


`Optimization/analysis/physics_diff.py`, git `76155eb48a`, config `dba94a5ecdc3`, physics hash `e6edcf3d1d1d`.

### Cumulative chain legacy → lim_exact (optimised R, bits/pulse)

| stage | L=0, n_X=1e+04 | L=50, n_X=1e+04 | L=100, n_X=1e+04 | L=150, n_X=1e+04 | L=0, n_X=1e+09 | L=50, n_X=1e+09 | L=100, n_X=1e+09 | L=150, n_X=1e+09 |
|---|---|---|---|---|---|---|---|---|
| 0 legacy (src/qkd switches) | 3.800e-04 | 3.543e-05 | 2.598e-06 | (-1.0e-25, l≤0) | 1.090e-02 | 1.052e-03 | 9.881e-05 | 5.855e-06 |
| 1 m_Z = total Z errors (Eq. 4) | 8.386e-04 | 7.770e-05 | 5.393e-06 | (-7.2e-27, l≤0) | 1.567e-02 | 1.522e-03 | 1.420e-04 | 1.006e-05 |
| 2 e_mis 0.01 -> 5e-3 | 1.233e-03 | 1.145e-04 | 8.199e-06 | 1.079e-09 | 1.911e-02 | 1.860e-03 | 1.736e-04 | 1.238e-05 |
| 3 p_ap 0 -> 0.04 in e_k | 2.812e-04 | 2.530e-05 | 1.221e-06 | (-4.5e-25, l≤0) | 8.966e-03 | 8.631e-04 | 7.894e-05 | 5.055e-06 |
| 4 R_k = D_k(1+p_ap) in counts/N | 3.148e-04 | 2.847e-05 | 1.471e-06 | (-3.4e-25, l≤0) | 9.670e-03 | 9.318e-04 | 8.565e-05 | 5.646e-06 |
| 5 eps_sec = kappa*l (kappa=1e-15) | 2.651e-04 | 2.374e-05 | 1.067e-06 | (-9.3e-25, l≤0) | 9.930e-03 | 9.570e-04 | 8.801e-05 | 5.861e-06 |
| 6 no clamp of n^-, m^- at 0 | 2.084e-04 | 1.883e-05 | 9.913e-07 | (-9.3e-25, l≤0) | 9.203e-03 | 8.897e-04 | 8.537e-05 | 5.861e-06 |
| 7 eta_ch (paper text) in e_mis term = lim_exact | 4.897e-08 | (-6.0e-25, l≤0) | (-2.5e-24, l≤0) | (-2.9e-08, l≤0) | 1.250e-03 | 9.616e-05 | 7.939e-06 | (-7.8e-27, l≤0) |

### Paper targets (15 % tolerance, never tuned)

| model | T1 R(0,1e9) [~1e-2] | T2 ratio [~1.75] | T3 L at R=1e-7, n_X=1e4 [~135 km] | horizon n_X=1e4 (km) |
|---|---|---|---|---|
| lim_exact | 1.2505e-03 (MISMATCH) | 2.734 (MISMATCH) | R < 1e-7 on the whole 60-180 km grid (MISMATCH) | None |
| lim_exact[emis_eta=sys] (diagnostic, NOT a preset) | 9.2025e-03 (ok) | 1.791 (ok) | 118.78 (ok) | 130.4409484853968 |
| legacy | 1.0900e-02 (ok) | 1.790 (ok) | 134.14 (ok) | 148.17417520377785 |

### Reading of the targets

- **`lim_exact` (literal paper text, η_ch in the e_mis term) fails all three paper targets.**
  - R(0, 1e9) is 1.25e-3, versus about 1e-2.
  - The 100 km ratio is 2.73, versus 1.75.
  - n_X = 1e4 never reaches 1e-7 between 60 and 180 km. Its horizon is about 19 km; the reference horizons are in `results/benchmark_lim_exact_*/RESULTS.md`.
- **The single switch η_ch → η_sys (diagnostic only, not a preset) meets all three within 15 %.** The other `lim_exact` settings are unchanged.
- This is strong evidence that "η_ch" in the paper's e_k is a typo for η_sys, or that the authors' simulation used η_sys.
- I have **not** changed `lim_exact`. It stays literal, as specified. Whether to adopt η_sys is your decision.
- The value R(0, 1e9) = 1.2505e-3 that you quoted as "Lim ≈ 1.25e-3 (ours 1.23e-3)" equals the literal-η_ch result.
- `legacy` also meets the three targets within 15 %, but it does so with e_mis = 0.01, p_ap = 0, ε_sec fixed and the m_Z bug.

