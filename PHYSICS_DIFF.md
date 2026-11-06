# Physics presets vs Lim et al. 2014: differences, unprinted choices, paper targets

Reference: C. C. W. Lim, M. Curty, N. Walenta, F. Xu, H. Zbinden, *Concise security bounds for practical
decoy-state quantum key distribution*, PRA 89, 022307 (2014), arXiv:1311.7129. The equations were read
from the arXiv LaTeX source: Eqs. (1)–(5) and the "Evaluation" section.

All numbers in this file come from generated outputs; none are typed by hand:
- the tables at the end are pasted verbatim from `results/physics_diff_*/PHYSICS_DIFF_TABLES.md`, written by `Optimization/analysis/physics_diff.py`;
- the deviations are the `\QKD…TargetDevPct` values in `results/NUMBERS.md`, written by `Optimization/report/make_numbers.py`.

## Presets

| preset | role | definition |
|---|---|---|
| **`lim`** (default) | Lim et al. 2014 finite-key decoy-state bounds, **with η_sys in the misalignment term** | `src/qkd_opt/physics_lim.py`. e_mis = 5e-3, p_ap = 4e-2, p_dc = 6e-7, η_Bob = 0.1, α = 0.2 dB/km, f_EC = 1.16, ε_cor = 1e-15, μ3 = 2e-4, ε_sec = κℓ with κ = 1e-15 (greatest fixed point), R_k = D_k(1+p_ap), m_Z = Σ_k m_{Z,k}, e_k = p_dc + e_mis[1 − exp(−η_sys k)] + p_ap D_k/2. |
| `lim_literal` | diagnostic | Same as `lim`, but with the printed **η_ch** in the e_mis term. |
| `legacy` | the physics of all earlier results | `physics_lim.py` with legacy switches: e_mis = 0.01, p_ap = 0, ε_sec = 1e-10 fixed, m_Z = Σ n_{X,k}, R_k = D_k, n⁻/m⁻ clamped at 0. Tested equal to `src/qkd` itself (preset `legacy_src`) to < 1e-8 relative in ℓ at interior points (`tests/test_qkd_opt_physics.py`). It does not depend on the `src/qkd` working tree. `src/qkd`'s extra degenerate-region clamps (γ ≤ 10, v/s ∈ [−10, 10], b sanitised) are not reproduced. |

**`lim` reproduces Lim et al. within ~10 %, not exactly.**
- Its deviations from the three numbers stated in the paper are `\QKDLimTargetROneDevPct`, `\QKDLimTargetRatioDevPct` and `\QKDLimTargetReachDevPct` (table below). The largest is the n_X = 1e4 reach.
- `lim_literal` (printed η_ch) misses all three by far.
- So η_ch in the printed e_k is treated as a probable typo, or as not what the authors simulated. This is an inference; the paper does not confirm it.

## Unprinted model choices (the paper does not specify these; our choice is stated)

1. **Expected-count model.** n_{X,k} = N q_x² p_k R_k and n_{Z,k} = N (1−q_x)² p_k R_k, with N fixed by Σ_k n_{X,k} = n_X. Likewise m_{X,k} = N q_x² p_k e_k and m_{Z,k} = N (1−q_x)² p_k e_k. Expected values are used as "observed" counts, with no sampling.
2. **e_obs.** The paper says "the average of the observed error rates in basis X". We use Σ_k m_{X,k}/n_X, i.e. the detection-weighted average.
3. **Transmittance in the misalignment term.** The printed text says η_ch; `lim` uses η_sys (see above).
4. **ε_sec = κℓ.** This is self-referential. We take the **greatest** self-consistent fixed point. ε = κ (ℓ ≤ 1) is always a trivial fixed point; when the non-trivial one disappears near the horizon, R(p) is discontinuous.
5. **The floor in Eq. (1)** is dropped. This continuous relaxation is needed for gradient-based optimisation.
6. **s_{X,0}, s_{Z,0} are clamped at ≥ 0.** The single-photon term is dropped if s_{X,1} ≤ 0 or s_{Z,1} ≤ 0.
7. **φ_X is capped at ½.** h is symmetric, so φ ≥ ½ yields no single-photon key. Likewise φ_X = ½ is used when b = v_{Z,1}/s_{Z,1} ≤ 0 (`physics_lim.py`, `b_ok`).
8. **n⁻, m⁻ are not clamped at 0** (as printed). Clamping would give a valid, tighter bound, as `legacy` does.
9. **Basis sifting.** Both parties choose X with probability q_x, giving factor q_x² (Z: (1−q_x)²). Double clicks are assigned randomly (the protocol text), which is implicit in D_k.
10. **Optimisation box** (not part of the bound): μ1 < 0.9, μ2 < 0.5, μ3 = 2e-4 fixed, and P_{μ1}, P_{μ2}, P_{μ3}, q_x ∈ (0, 1). The paper does not state its search box.
11. **Dark-count convention.** D_k = 1 − (1 − 2p_dc) exp(−η_sys k), i.e. two detectors (as printed).

## Differences between `src/qkd` (legacy physics) and the paper

`src/qkd` line numbers refer to commit `d185d8f` (the author's refactor, committed as found).

| # | item | paper | src/qkd | note |
|---|---|---|---|---|
| (i) | transmittance in the e_mis term of e_k | η_ch (printed) | η_sys (`physics.py:204`, passed at `model.py:33`) | Coincides with `lim` (η_sys) and differs from `lim_literal`. |
| (ii) | m_Z in Eq. (4) | Σ_k m_{Z,k}, the Z-basis **errors** | `model.py:84` sums the **n_X detections** | **A bug.** It inflates √(m_Z/2·ln(21/ε)) and therefore v_{Z,1} and φ_X. Chain stage 1. |
| (iii) | 6 log₂(21/ε_sec) | as in Eq. (1) | `physics.py:279` matches | `physics.py:133` clips ε ≥ 1e-10 inside the √ term only. This is consistent while ε = 1e-10 is fixed. |
| (iv) | ε_sec | κℓ, κ = 1e-15 | 1e-10, fixed | Chain stage 5. |
| (v) | afterpulsing in the detection rate | R_k = D_k(1+p_ap) | D_k (`physics.py:69-71, 99-105`) | Chain stage 4. |
| (vi) | p_ap, e_mis | 4e-2, 5e-3 | 0, 0.01 | Chain stages 2 and 3. |
| (vii) | n⁻, m⁻ | not clamped | clamped at 0 (`physics.py:143, 191`) | A valid, tighter bound. Chain stage 6 lowers R where n_k < δ. |
| (viii) | s_0, s_1 | lower bounds | clamped at 0, plus 1e-15 in the S_0 denominator | Same treatment as `physics_lim`. |
| (ix) | γ, φ | Eq. (5) | b sanitised, γ ≤ 10, v/s ∈ [−10, 10], φ ≤ ½; γ gets v/(s+1e-12) while Φ uses v/(s+1e-15) | Only active in degenerate regions. |
| (x) | floor in Eq. (1) | ⌊…⌋ | none | Same as `physics_lim`. |

## Generated tables

_(Pasted verbatim from the generated `PHYSICS_DIFF_TABLES.md`, and from `results/NUMBERS.md` for the deviations.)_
