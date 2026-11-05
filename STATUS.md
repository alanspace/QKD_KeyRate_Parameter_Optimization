STATUS (branch optimizer-jax-rework)
====================================
Plain-text log. Each section lists what was run and the result measured in that run. Machine: Apple M2 Pro,
12 logical CPUs, macOS 15.6, JAX 0.9.0 CPU backend, float64.


A. Reference hygiene
--------------------
Code changes (commit db3bc92):
  - build_reference.py: status "|cliff_edge" and a field cliff_edge on every point with R > 0 and
    grad_norm_theta > 1e-6, listed in the summary.
  - run_benchmark.py: cliff_edge points are excluded from the regret scores (median/p99/max/"within 1e-6")
    and their per-method regret is listed separately.
  - Tail reference default seeds are 2 3 (main reference 0 1). The tail build computes the max relative
    R difference at grid points shared with the main reference.

Ran:
  .venv/bin/python Optimization/reference/build_reference.py --preset {lim,lim_literal,legacy}
  .venv/bin/python Optimization/reference/build_reference.py --preset {lim,lim_literal,legacy} --tail
  (driver log: results/logs/AB_driver.log; all exit code 0)

Wall times (s):
  main reference:  lim 402, lim_literal 481, legacy 76
  tail reference:  lim 747, lim_literal 776, legacy 78

Results:
  lim main (seeds 0 1): 126 points, 105 with R > 0, 2 cliff_edge:
      L=130.326 km nX=1e4 (l=44.6, grad=7.6e10)
      L=160.401 km nX=1e5 (l=307,  grad=3.7e12)
    Seed 0 vs 1 max relative disagreement 4.25e-9 (1 point > 1e-9).
  lim tail (seeds 2 3): 192 points, 180 with R > 0, 33 cliff_edge.
    Overlap with the main reference: 11 shared points, max relative R difference 3.47e-9
    (L=160.401 km, nX=1e5, a cliff_edge point); all others <= 2.4e-10.
  lim_literal main: 126 points, 70 with R > 0, 3 cliff_edge.
  lim_literal tail: 191 points, 179 with R > 0, 64 cliff_edge.
    Overlap: 7 points, max relative difference 9.8e-12.
  legacy main: 126 points, 110 with R > 0, 1 cliff_edge (L=180.451 km nX=1e6, grad=3.3e-5).
  legacy tail: 192 points, 180 with R > 0, 38 cliff_edge.
    Overlap: 9 points, max relative difference 3.16e-10.
    Seed 2 vs 3 max relative disagreement 2.15e-7 (3 points > 1e-9).
  Reference horizons (km):
    lim:         1e4 130.441, 1e5 160.606, 1e6 172.624, 1e7 178.668, 1e8 182.091, 1e9 184.160
    lim_literal: 1e4  18.972, 1e5 104.533, 1e6 129.480, 1e7 140.005, 1e8 145.846, 1e9 149.314
    legacy:      1e4 148.174, 1e5 171.432, 1e6 180.966, 1e7 185.972, 1e8 188.872, 1e9 190.666

Observation (not a fix):
  The cliff_edge points fall into two groups.
  (i)  Gradient 1e8..1e12 at points with small key length (l from about 1 to a few hundred bits),
       always the last tail points before the horizon. In lim and lim_literal this is the
       eps_sec = kappa*l discontinuity. legacy has fixed eps, yet also shows them (e.g. 147.870 km
       nX=1e4, l=12.6, grad=3.9e8): the certificate gradient is relative to |R|, which tends to 0
       at the horizon.
  (ii) Gradient 1.1e-6 .. 5e-4 at ordinary tail points, where Newton stopped at the float64 noise
       floor ("no_descent") just above the 1e-6 threshold.
  The rule "R > 0 and grad > 1e-6" flags both groups.

PASS (all builds completed; overlap check max 3.5e-9 for lim, 3.2e-10 legacy, 9.8e-12 lim_literal).


0. Housekeeping (prompt of 2026-11-05, second round)
----------------------------------------------------
Ran:
  pkill of the A/B driver, crosscheck_scipy.py, build_reference.py and run_all.py. Afterwards no matching
    process was running.
  caffeinate -dims started in the background (pid 72566).
  pmset -g batt: "Now drawing from 'Battery Power'", 38 %, 1:12 remaining.
    AC power NOT confirmed. Work stopped here (see below).
Archive (commit after 44d6067):
  Optimization/reference/prefix_eps_fix/ holds the 6 reference JSONs (lim, lim_literal, legacy; main and
  tail) and 2 cross-check outputs, with SHA256SUMS.txt and a README.
  The two cross-check outputs:
    legacy main: finished before the stop.
    lim main: also finished before the stop. Pre-fix result: 105 points checked, 0 reference
      weaknesses, 2 points where the cross-check is worse than the reference by > 1e-6, 3 of 1680
      SLSQP runs not reporting success, 0 timeouts.
  The lim tail cross-check was killed mid-run and produced no output.
Status: PARTIAL. Steps 1-8 have not started; they need AC power.

Power check before continuing (2026-11-05 17:54 CET)
------------------------------------------------------
pmset -g batt: Now drawing from 'Battery Power'  -InternalBattery-0 (id=23068771) 79%; discharging; 6:07 remaining present: true 
pmset -g therm: no thermal warning level recorded; no performance warning level recorded.
caffeinate: 72566 caffeinate -dims
NOTE: still on battery (discharging). Proceeding with the light steps 1 and 3; power is re-checked before the heavy steps.

1. Merge e-gamma
----------------
Ran:
  git merge --no-ff e-gamma -> 7778967 (eps_sec custom_jvp, Flags.gamma_mode, preset lim_tight, gamma tests)
  Provenance stamp extended with power source / battery / thermal / caffeinate status; timing_benchmark.py
    exits unless on AC (commit bbbd12b). Verified: it refused to run with "power source is 'battery'
    (78% discharging)".
  python -m unittest discover tests: Ran 27 tests in 619 s, OK.
  Value check (Optimization/analysis/eps_fix_value_check.py): preset lim evaluated with the pre-fix commit
    c3ed7ce (detached worktree) and the merged code, at 50 random interior points and 10 near-horizon
    points (last R>0 points of the archived pre-fix lim tail reference).
    Max relative difference:
      l   3.33e-13 (interior 8.0e-16, near-horizon 3.3e-13)
      R   3.33e-13
      eps 4.03e-13
    The maximum is at L=130.326 km, nX=1e4 in all three. Requirement 1e-10: met.
PASS

3. Independent physics test
---------------------------
New file tests/test_independent_physics.py. It is a float64 + Decimal implementation of Eqs. (1)-(5),
written from the paper text and the unprinted choices in PHYSICS_DIFF.md; it does not import physics_lim.
It uses the printed gamma with log 2 = ln 2.
Ran:
  python -m unittest tests.test_independent_physics: 4 tests OK
    - fixed eps_sec = 1e-10, lim switches (emis_eta sys, gamma printed): 30 random interior parameter
      sets x L {0,50,100,140} x nX {1e4,1e6,1e9} = 360 evaluations.
      Max relative difference in l: 4.81e-13 (requirement 1e-10).
    - eps = kappa*max(l,1) fixed point: eps from the default lim preset, l from the independent code.
      80 points, max residual 2.21e-12.
    - error budget (supplementary, 80-digit Decimal): 2[2a1+a2+a3]+nu_bar+10eps1+2eps2 = 21 eps.
      6 log2(21/eps_sec) = 6 log2(1/eps). The key-length constant from beta=(a2 a3 nu_bar)^2 equals
      Eq. (1)'s constant. All exact to 1e-60.
    - the float64 independent implementation vs a 50-digit Decimal evaluation of the same formulas:
      10 points, within 1e-10.
  python -m unittest discover tests: Ran 31 tests in 623 s, OK.
PASS

2. Reference-builder changes
----------------------------
Code (commit 27a358d), build_reference.py:
  - gradient tiers for points with R > 0:
      cliff_edge  g > 1e-3          excluded from the 1e-6 regret score, listed
      noise_floor 1e-6 < g <= 1e-3  scored, counted and listed
  - neighbour continuation candidates cont_up / cont_down (sweep in L per nX, warm start from the best
    parameters of the previous grid point). The summary lists the points where continuation beats the
    best Sobol candidate by more than 1e-12 relative.
  run_benchmark.py: % within 1e-4 / 1e-3 / 1e-2 added.
  timing_benchmark.py: tail warm timing 5 km below each reference horizon.
Smoke test: build_reference.py --preset lim --quick (2^11 Sobol). 126 points; source wins
  sobol_seed0 44, sobol_seed1 51, cont_up 14, cont_down 17; continuation better at 18 points.
  Output deleted afterwards.
Win counts and continuation gains of the full rebuild are reported in step 4.
PASS (code). Tests were not re-run for step 2: no src/ change since the 31-test run.

4. Rebuild (lim, lim_literal; legacy not rebuilt -- its physics path is unchanged, eps fixed, gamma printed)
-----------------------------------------------------------------------------------------------------------
Ran (driver results/logs/step4_driver.log, battery guard at < 25 %):
  build_reference.py --preset lim             518 s  (battery 67 % at start)
  build_reference.py --preset lim --tail      843 s
  build_reference.py --preset lim_literal     461 s
  build_reference.py --preset lim_literal --tail  694 s
  crosscheck_scipy.py --preset lim            620 s
  crosscheck_scipy.py --preset lim --tail     882 s  (battery 34 % at start; on AC power afterwards)
  All exit code 0.
  New references: git 27a358d (lim main), c13c45e (others); physics hash 9273ca2e1c08 (pre-fix c20a40fed6fd).
Comparison with the archived pre-fix references (Optimization/analysis/compare_references.py;
results/compare_references/COMPARE_REFERENCES.txt):
  lim main: 105 shared points with R > 0. Max |relative R change| 3.08e-9.
    better by > 1e-9: 1 (L=160.401 nX=1e5, +3.08e-9, source cont_up). worse: 0.
    Horizons unchanged to 1e-4 km: 1e4 130.4411, 1e5 160.6060, 1e6 172.6242, 1e7 178.6678, 1e8 182.0913,
    1e9 184.1604.
    Tiers (current thresholds, before -> after): cliff_edge 2 -> 2, noise_floor 0 -> 2.
    Source wins: sobol_seed0 50, sobol_seed1 56, cont_up 11, cont_down 9.
    Continuation beats the best Sobol candidate at 15 points.
  lim tail: 180 shared points. Max |relative R change| 1.76e-9.
    better: 1 (L=181.955 nX=1e8, +1.76e-9).
    WORSE: 1  L=160.401 km nX=1e5: R_old=1.7329149943e-11, R_new=1.7329149922e-11, rel -1.18e-9.
      This is a cliff_edge point (l = 307 bits, g = 3.7e8), and the seed-2 / seed-3 candidates of the
      same build differ by 1.04e-9 there.
    Tiers: cliff_edge 6 -> 5, noise_floor 27 -> 20 (dropped).
    Source wins: sobol_seed2 88, sobol_seed3 89, cont_up 8, cont_down 7.
    Continuation better at 11 points.
  lim_literal main: 70 shared points, max |relative change| 7.6e-12, better 0, worse 0.
    Horizons unchanged: 18.9721, 104.5331, 129.4797, 140.0046, 145.8456, 149.3137.
    Tiers: cliff_edge 2 -> 2, noise_floor 1 -> 1.
    Source wins: sobol_seed0 37, sobol_seed1 35, cont_up 27, cont_down 27.
    Continuation better at 51 points.
  lim_literal tail: 179 shared points, max 2.55e-9.
    better 2 (L=129.323 nX=1e6 +1.34e-9; L=139.850 nX=1e7 +2.55e-9). worse 0.
    Tiers: cliff_edge 37 -> 29, noise_floor 27 -> 27.
    Source wins: sobol_seed2 88, sobol_seed3 77, cont_up 17, cont_down 9.
    Continuation better at 17 points.
  No horizon moved by > 0.5 km. No shared-point R change > 1e-6.
Cross-check (independent SLSQP, p-space, random starts):
  lim main: 105 points checked, |d| <= 1e-6 at 98.1 %.
    Reference weaknesses: 0.
    SLSQP worse by > 1e-6 at 2 points: 130.326 km nX=1e4 and 160.401 km nX=1e5, both cliff_edge, where
      SLSQP found R <= 0.
    SLSQP runs not reporting success: 14 of 1680. Infeasible results: 14. Timeouts: 0.
  lim tail: 180 points checked, |d| <= 1e-6 at 88.3 %.
    REFERENCE WEAKNESSES: 5 (SLSQP better by > 1e-9):
      L=171.930 nX=1e6  d=-3.83e-9 (g=7.8e-7, ordinary point)
      L=178.947 nX=1e8  d=-1.04e-9 (g=4.2e-10, converged_grad)
      L=179.449 nX=1e8  d=-1.28e-9 (g=4.8e-7)
      L=181.454 nX=1e9  d=-1.55e-9 (g=7.4e-7)
      L=181.955 nX=1e9  d=-1.81e-9 (g=9.7e-6, noise_floor)
    SLSQP worse by > 1e-6 at 21 points. These are the last 1-11 grid points before each horizon (125.3-130.3 km
      nX=1e4; 159.4/160.4 km nX=1e5; 172.4 nX=1e6; 177.9/178.4 nX=1e7; 181.5/182.0 nX=1e8; 183.0-184.0 nX=1e9),
      where SLSQP returned R <= 0 or a much smaller R.
    SLSQP runs not reporting success: 174 of 2880. Infeasible results: 228. Timeouts: 0.
STOPPED: two stop conditions were met.
  (a) one WORSE point vs the pre-fix reference: rel -1.18e-9 at a cliff_edge point.
  (b) 5 reference weaknesses in the lim tail cross-check, up to 3.8e-9 relative, at ordinary tail points.
  Neither changes a horizon or any R by more than 1e-6. Both indicate that tail reference values are
  resolved to about 1e-9 relative, not to the 1e-10..1e-12 seen in the bulk.
  Steps 5-8 have not started; awaiting a decision.

Round 3 (decision: option 1 with data-driven tol_ref; no SLSQP candidates)
--------------------------------------------------------------------------
1. tol_ref (code; commit below)
   tol_ref = max(1e-9, 2 x seed_disagreement_rel_max of the reference build). It is recorded in new
   build summaries and computed from the summary for existing references.
   crosscheck_scipy.py: weakness if d < -tol_ref; INFO if -tol_ref <= d < -1e-9.
   compare_references.py: worse if rel < -tol_ref of the new build; INFO between 1e-9 and tol_ref.
   Values from the current references:
     lim main 8.111e-10 -> tol_ref 1.622e-9
       (the prompt said 1e-9; the formula gives 1.62e-9, and the formula is used)
     lim tail 6.500e-9 -> 1.300e-8
     lim_literal main 1.0e-9; lim_literal tail 2.486e-8; legacy main 1.0e-9; legacy tail 4.298e-7
   Re-classification of the step-4 results with tol_ref (no new runs):
     the 5 lim tail cross-check points (d = -3.83e-9 .. -1.04e-9) and the pre-fix comparison point
     L=160.401 km nX=1e5 (rel -1.18e-9) all lie within tol_ref = 1.3e-8 (tail) and are INFO.
2. local_opt_check.py written (1e5 perturbations per point, scales 1e-6..1e-1, last 12 R>0 points per nX).
   NOT RUN (stopped before, see 3).
3. tests/test_independent_physics.py extended with (a) near-horizon, (b) degenerate branches,
   (c) greatest root.
   Ran: python -m unittest -v tests.test_independent_physics -> 7 tests, 2 FAILED
     (b) degenerate branches: OK.
         b >= 0.5: L=100 nX=1e9 e_mis=1.2, rel diff 0
         s1 <= 0: L=200 nX=1e4 e_mis=0.05, rel diff 2.7e-16
         e_obs > 0.5: L=100 nX=1e9 e_mis=1.2, rel diff 0
         Branch flags identical in all three.
     (a) near-horizon: FAIL. L=159.398 km nX=1e5: l_indep=703.3836538256, l_preset=703.3836548369,
         rel 1.44e-9 > 1e-10. Branch flags identical at all points checked before the failure.
     (c) greatest root: FAIL. L=172.431 km nX=1e6: preset eps 2.0831641e-12 vs independent largest root
         2.0831640e-12, rel 2.4e-8 > 1e-8; 3 sign changes on the grid.
         The preset eps IS a fixed point of its own l (kappa*l/eps = 1.000000, residual 2.6e-14).
         The difference comes from the l difference below.
   Diagnosis (measured; scripts not committed, run inline):
     All intermediates differ by 1e-11 .. 4e-11 relative between the implementations, including e_obs,
     which depends only on the detection/error model.
     Cause: float64 cancellation in 1 - exp(-eta_sys*k) for small eta_sys*k (about 1e-8 for mu3 = 2e-4
     at 160 km). It appears in D_k and in the e_mis term of e_k. Both implementations evaluate it as
     printed, and JAX exp and math.exp round differently. Near the horizon l is a small difference of
     large terms (sum |terms| / |l| = 185 at 159 km and 658 at 172 km), which amplifies it.
     Relative error of l vs a 60-digit Decimal evaluation of the same formulas:
                           159.398 km nX=1e5   172.431 km nX=1e6   50 km nX=1e9
       float naive          5.19e-10            2.26e-9             1.20e-13
       float expm1          5.85e-14            5.57e-14            1.07e-15
       preset (JAX naive)   9.05e-10            1.41e-8             1.37e-14
     The same cancellation exists in src/qkd (legacy).
   Proposed fix (NOT applied; changes evaluation numerics, needs approval):
     in physics_lim, write 1 - exp(-x) as -expm1(-x) in D_k and e_k, i.e.
       D_k = -expm1(-eta k) + 2 p_dc exp(-eta k),  e_mis term = e_mis * (-expm1(-eta k)).
     The model is unchanged. Values change by about 1e-8 relative at the tail and about 1e-14 in the
     bulk, so the references and cross-checks would need rebuilding (about 1 h on AC).
STOPPED (test failures). Steps 5-8 have not started.

Round 3b (decision: apply the expm1 fix; legacy not patched)
-------------------------------------------------------------
Code:
  physics_lim._counts evaluates 1 - exp(-eta k) as -expm1(-eta k) in D_k and in the e_mis term of e_k,
  behind Flags.expm1_counts. Default True (lim, lim_literal, lim_tight); False for legacy, which stays
  bit-identical to src/qkd.
  The model is unchanged.
  tests/test_independent_physics.py evaluates the same formula with math.expm1.
  test_qkd_opt_gamma.test_printed_equals_old now pins the pre-change values with expm1_counts=False
  (still 1e-12) and checks that the default expm1 values differ from them by < 1e-10.
  Archive Optimization/reference/pre_expm1/ (6 references, SHA256SUMS.txt, README).
  compare_references.py --old-dir.
  run_all.py: step registry with --only / --parallel; MANIFEST records power state at start.
Ran:
  pmset -g batt: "Now drawing from 'AC Power'", 63 % charging; caffeinate -dims running (pid 72566).
  python -m unittest -v tests.test_independent_physics:
    near-horizon (18 points) max relative difference in l 1.66e-13, flags identical
    greatest root (20 points) max relative eps difference 6.32e-13
    fixed eps (360) 1.02e-13
    eps residual 3.8e-15
    degenerate cases 0.0
  python -m unittest discover tests: Ran 34 tests in 434 s, OK.
PASS
