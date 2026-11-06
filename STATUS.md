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

4 (redo after the expm1 fix). Rebuild, comparison, cross-check, local optimality
--------------------------------------------------------------------------------
Ran (python Optimization/run_all.py --only ...; results/MANIFEST.json; on AC power):
  reference:lim 302 s, reference_tail:lim 435 s, reference:lim_literal 297 s, reference_tail:lim_literal 485 s,
  crosscheck:lim 423 s, crosscheck_tail:lim 718 s, compare_references 0.1 s, local_opt_check:lim 26 s.
  All exit code 0. The references are stamped git 9b2938e, physics hash 9564691bc5b0, power AC.
Seed-to-seed max relative disagreement (expm1 / pre-expm1):
  lim main 3.30e-13 / 8.11e-10
  lim tail 7.56e-13 / 6.50e-9
  lim_literal main 4.6e-14
  lim_literal tail 1.1e-12
  tol_ref is therefore 1e-9 (the floor) for all four builds.
Tiers (current thresholds, after):
  lim main:         cliff_edge 2,  noise_floor 0 (before 2)
  lim tail:         cliff_edge 5,  noise_floor 1 (before 20)
  lim_literal main: cliff_edge 2,  noise_floor 0
  lim_literal tail: cliff_edge 33, noise_floor 5 (before 27)
  The noise_floor count dropped in all four builds.
Continuation beats the best Sobol candidate at: lim main 14, lim tail 12, lim_literal main 54,
  lim_literal tail 13 points.
Horizons: all 12 (lim and lim_literal, six nX each) are unchanged to 1e-4 km vs pre-expm1.
  lim: 130.4411 160.6060 172.6242 178.6678 182.0913 184.1604
  lim_literal: 18.9721 104.5331 129.4797 140.0046 145.8456 149.3137
Comparison with the pre-expm1 references (compare_references --old-dir pre_expm1): the stored R decreased at
  5 main and 37 tail lim points, and at 39 lim_literal tail points. All decreases are in the last ~3-15 km before
  each horizon; the largest is -4.46e-8 (L=181.955 nX=1e8). No horizon changed.
  Explanation, measured with Optimization/analysis/compare_same_objective.py: re-evaluating the OLD reference
  parameters with the current (expm1) objective reproduces these decreases.
    Evaluation change of the old params: lim tail min -4.46e-8, lim_literal tail min -2.92e-8.
    The new references are then never worse than the old parameters:
      gap (new ref - current R(old params)) min -1.8e-16 (lim main), +1.4e-15 (lim tail),
      +4.2e-15 / +7.8e-15 (lim_literal main / tail).
      WORSE 0 in all four builds.
  So the decrease is the removal of the float64 overestimate, not an optimisation regression.
  This matches the 60-digit Decimal check: the pre-expm1 l at 172.431 km nX=1e6 was 1.41e-8 too high.
Cross-check (independent SLSQP):
  lim main: 105 points, |d| <= 1e-6 at 98.1 %, min d -2.66e-15.
    Reference weaknesses (> tol_ref) 0. INFO 0.
    SLSQP worse by > 1e-6: 2 points (130.3 km nX=1e4, 160.4 km nX=1e5, both cliff_edge).
    SLSQP not reporting success: 11 of 1680. Infeasible results 12. Timeouts 0.
  lim tail: 180 points, |d| <= 1e-6 at 87.8 %, min d -2.01e-14.
    Reference weaknesses 0. INFO 0.
    SLSQP worse by > 1e-6 at 22 points, all within the last ~5 km before each horizon (L km / nX):
      125.3-130.3 / 1e4 (11 points), 159.4 159.9 160.4 / 1e5, 172.4 / 1e6, 177.9 178.4 / 1e7,
      181.5 182.0 / 1e8, 183.0 183.5 184.0 / 1e9.
    SLSQP not reporting success: 167 of 2880. Infeasible results 236. Timeouts 0.
Local-optimality perturbation check (lim tail, last 12 R>0 points per nX = 72 points, 1e5 perturbations each,
  scales 1e-6..1e-1): max relative improvement 4.86e-13; points improving by > tol_ref: 0. Wall 25 s.
PASS (no worse point under the same objective, 0 reference weaknesses, no horizon change).

5. Tasks C, D, E (preset lim, expm1 evaluation; all on AC; via run_all, results/MANIFEST.json)
------------------------------------------------------------------------------------------
5C cliff test (results/cliff_test_abf88e5525_bf6fb332afdd, 133 s):
  nX=1e4, L=128..131 km step 0.25. Method 1 = greatest-fixed-point objective; method 2 = outer fixed-eps loop.
    Method 1 gives larger l at every grid point with l > 0 under both methods
      (e.g. 128.00 km: l 120.15 vs 89.38; 129.75 km: 64.53 vs 24.65).
    Method 2 did not converge within 80 outer iterations at 5 of 13 points (128.00, 128.50, 128.75, 129.25,
      129.75 km).
    Horizon method 1: 130.4458 km. Method 2: 129.8708 km. |difference| 0.575 km -> differs by > 0.5 km: YES.
  nX=1e9, L=182..184.5 km step 0.25: all method-2 runs converged; method 1 again gives larger l.
    Horizon method 1: 184.1641 km. Method 2: 184.1448 km. |difference| 0.019 km -> no.
  Default unchanged (as instructed). The method-1 horizon agrees with the reference horizon
  (130.4411 / 184.1604 km) to 0.005 km.
5D dense scan (results/dense_scan_abf88e5525_91f81e1652a5, 759 s, run in parallel with 5E):
  lim, L 100 km .. horizon+1 at 1 km, all six nX, 2^17 Sobol, top-64, seeds 2 3.
  Jumps > 0.05 (normalised) per nX: 1e4 0, 1e5 3, 1e6 5, 1e7 5, 1e8 5, 1e9 5. All lie in the last ~5 km
    before each horizon.
  The script's R-continuity classification marks most of them "R discontinuous / no branch crossing".
    That classification is not reliable near the horizon: log R -> -inf, and the u-space persistence test
    (0.02) is smaller than the per-km drift of a continuous optimum.
  Refinement (Optimization/analysis/jump_refine.py, 0.1 km warm continuation + cold 2^14 per step):
    for every jump below the horizon the largest 0.1 km step is 11-12 % of the 1 km jump, i.e. continuous
    drift. The pairs that cross the horizon reach 16-54 %.
    -> lim has NO branch switch between 100 km and the horizon for any nX.
  Legacy comparison at nX=1e9 (dense scan 140..160 km, legacy preset): one jump at 154->155 km, classified
    branch crossing with R continuous.
    Stored legacy runs, same window: legacy_Hessian 148.37->148.87 (jump 0.230); legacy_DA 5 jumps
    (150.9..157.4); legacy_Hessian_FoR 5 jumps (148.4..151.9).
  The 148-155 km legacy events have no counterpart in lim.
5E gamma report (results/gamma_report_abf88e5525_5d1dbbb3d1df, 3778 s):
  Horizons (km), nX 1e4 / 1e5 / 1e6 / 1e7 / 1e8 / 1e9:
    printed 130.44 / 160.61 / 172.62 / 178.66 / 182.09 / 184.16
    exact   132.35 / 161.31 / 172.83 / 178.73 / 182.11 / 184.17
    max     identical to printed: printed gamma >= exact theta at every point checked
  R(0,1e9): printed 9.2025e-3, exact 9.2252e-3.
  R(100,1e9)/R(100,1e7): printed 1.7911, exact 1.7687.
  T3 (horizon nX=1e4, paper ~135 km): printed 130.44 (-3 %), exact 132.35 (-2 %).
  T3b (R = 1e-7 crossing): printed 118.78 km, exact 121.07 km.
  Printed gamma / exact theta at the 105 lim reference operating points: min 1.283, median 1.413, max 1.442.
physics_diff (results/physics_diff_a419cc7422_dba94a5ecdc3, 522 s; chain now has the expm1 stage):
  Targets: lim T1 9.2025e-3, T2 1.791, T3 horizon 130.44 km, T3b 118.78 km, all ok at 15 %.
    lim_literal: MISMATCH on all.
    legacy: ok on all (148.17 km horizon).
  The expm1 stage changes no chain horizon.
PASS (stop conditions: none. The 0.575 km difference in 5C is a requested comparison between two methods,
  not a change of a previously reported horizon.)

6. Benchmark and timing (all on AC)
-----------------------------------
Benchmarks:
  results/benchmark_lim_0c8f47a1f0_530f6d58460c (207 s; rerun after fixing the cross-check matching bug)
  results/benchmark_lim_literal_dfd7d80292_e98ec9129185 (220 s)
  results/benchmark_legacy_dfd7d80292_9eccc560dfd9 (65 s)
  cliff_edge points excluded, noise_floor included.
  lim bulk (103 points):
    new_cold:     100 % within 1e-6; max 9.1e-14
    new_warm:     100 %; max 1.0e-13
    lookup table: 81.6 / 96.1 / 96.1 / 98.1 % within 1e-6 / 1e-4 / 1e-3 / 1e-2;
                  median 1.36e-9, p99 1.27e-2, max 2.17e-2
  lim tail (175 points):
    new_cold:     100 %; max 6.5e-13
    new_warm:     100 %; max 9.5e-13
    lookup table: 24.6 / 68.0 / 82.3 / 93.7 %; median 1.51e-5, p99 18.2, max 23.1
  lim horizons, new_warm vs reference: within 0.008 km at all nX.
  lim_literal tail (146 points): new_cold 97.9 % within 1e-6 (max 8.7e-2); new_warm 98.6 % (max 5.3e-2).
    See open issues.
  legacy bulk (109 points):
    new_cold/new_warm 100 %
    old DA 3.7 %
    old Hessian 96.3 %
    old Hessian_FoR 81.7 %
  legacy tail (142 points):
    new_warm 100 %
    new_cold 99.3 % (1 point max 0.114)
    old DA 0 %
    old Hessian 95.8 %
    old Hessian_FoR 0 %
  Benchmark timing (lim): cold batched 40.0 ms/point (compile 38.4 s); warm sweep 4.27 ms/point
    (compile 80.9 s).
Timing (results/timing_lim_0c8f47a1f0_a412ba8ec55e, 180 s), run last on an idle machine:
  Before: 0 other Python jobs; pmset AC Power; no thermal or performance warning; caffeinate running.
  The stamp records power AC, 52 % charging.
  Old DA algorithm, archived settings on the lim objective: 5.8 / 11.7 / 17.6 s per point
    (110 / 150 / 190 km, nX=1e9), 8.8 / 11.7 s (130 / 145 km, nX=1e4); 116-118 us per evaluation.
  Old L-BFGS-B branch: 1-10 ms per point.
  JAX warm, batch 1, L=100 km nX=1e9:
    device-resident median 1199 us (p90 1322)
    end-to-end 1466 us
  JAX warm, batch 1, tail (5 km below each horizon, 6 nX):
    device-resident medians 1654-2554 us
    end-to-end 2024-2903 us
    3-5 Newton iterations
  Lookup table: device 7.7 us, end-to-end 10.0 us.
  NN: not measured.
PASS

7. Figures (results/figures/900d9f88a8_58c9f38705cd/)
------------------------------------------------------------
Nine figures (fig4 has two n_X panels -> 10 PNG files, 200 dpi) plus FIGURES.md (caption and read-off
number each) and figures.json. Every PNG was opened and checked. Fixes made, each followed by
regenerating and re-checking:
  fig2: legend covered the lim_literal curve in the n_X=1e4 panel -> moved to the upper right.
  fig4: x-range cut off the legacy runs at lim horizon + 3 km -> extended to all series.
        Dense-scan lines relabelled ">0.05 change per km (continuous at 0.1 km)".
  fig5: showed only a 1e-6 line -> both tiers drawn (1e-3 cliff_edge, 1e-6 noise_floor; noise_floor points
        coloured); legend overlapped the stamp -> moved.
  fig6: legend covered the new-optimizer curves -> moved below the panels; then cleared the stamp.
  fig8: tail warm timing row was missing -> added.
  FIGURES.md read-offs:
    fig4 now separates branch switches below the horizon (none for 1e4 and 1e9) from those beyond it
      (objective l/n_X).
    fig3 horizons rounded.
  fig1, fig3, fig7, fig9: no problems found.
PASS

8. Finish
---------
  python -m unittest discover tests (via run_all unittest:all): Ran 34 tests in 423 s, OK.
  python Optimization/report/make_numbers.py (via run_all numbers:all):
    wrote 127 values to result_report/generated_numbers.tex and results/NUMBERS.md from results/MANIFEST.json.
    Report prose not edited.
  apply_report_revision.py NOT applied (not requested this round).
PASS

Open issues
-----------
1. Silent small parameter step in the lim warm sweep at nX=1e9, 77.694 -> 78.195 km.
   mu1 0.43918 -> 0.43850 and P_mu1 0.75799 -> 0.75578 (about 0.002 normalised); R smooth.
   No event was logged: the warm local solve moved between two global checks.
   "Log every branch switch" is therefore not strictly met for small steps that need no global check.
   Proposed: log a "drift_step" event when |dp_norm|_inf > 1e-3 between consecutive warm solves.
2. lim_literal tail: new_cold / new_warm miss the reference by up to 8.7e-2 / 5.3e-2 at 3 / 2 of 146 tail
   points (diagnostic preset; points not yet listed or investigated).
3. legacy tail: new_cold has one point with regret 0.114.
   The legacy tail reference was built before tol_ref / continuation / the expm1 work (legacy is unchanged by
   expm1) and was not rebuilt this round.
4. 5C: the outer fixed-eps loop does not converge at 5/13 points and gives a 0.575 km shorter horizon at
   nX=1e4. Method 1 (default) gives larger l at every point and matches the reference.
5. dense_scan.py's R-continuity / persistence classification is unreliable near the horizon; jump_refine.py
   should replace or complement it.
6. analyses.py (sensitivity, robustness, extrapolation) was not re-run after the expm1 fix.
   make_numbers therefore has no Drift* macros, and the report's regenerated-results section would not
   compile until it is re-run.
7. The legacy references (main, tail) were not rebuilt with the continuation candidates; their tol_ref is
   1e-9 / 4.3e-7.
8. timing_benchmark prints "NN surrogate not measured (torch not installed)". Torch is installed in
   .venv-nn; the NN was not retrained, so there is still nothing to time.


9. Final report and audit (2026-11-06)
--------------------------------------
result_report/final_report.tex written (supersedes results_report.tex, the 2026-03 slides and the 2025 thesis
numbers). Every \QKD number is imported from generated_numbers.tex; typed numbers carry a dagger.
An independent read-only audit (separate agent, against result files and code) found 2 wrong statements and
~23 imprecise ones. All were corrected in the report. The substantive one:
  The lim warm sweep (results/benchmark_lim_0c8f47a1f0_530f6d58460c/summary.json, sweep_events) logged two
  branch switches BELOW the horizon, objective R:
    nX=1e5 at 135.34 km, rel_gain 7.7 %, u_dist 0.068, backfill crossover 130.3-130.8 km
    nX=1e6 at 120.30 km, rel_gain 0.6 %, u_dist 0.029, backfill crossover 118.3-118.8 km
  Section 5D above ("lim has NO branch switch between 100 km and the horizon") is therefore WRONG for nX=1e5
  and 1e6. It holds for nX=1e4, 1e7, 1e8, 1e9. The dense scan (threshold 0.05) did not flag either switch.
  Benchmark scores are unaffected (backfill puts the sweep on the better branch; 100 % within 1e-6).
Open issue 9 (new): characterise both optima at these crossovers, and make dense_scan sensitive to switches of
  this size.


10. QML surrogate study, branch qml-surrogate (2026-11-06)
-----------------------------------------------------------
Worktree ../QKD_qml_surrogate_wt, created from main at e796420.
Preregistration: qml/PREREGISTRATION.md, committed 6867b44 before any training; Amendments 1-3.

Environment .venv-qml (Python 3.13.3):
  pennylane 0.45.1, pennylane_lightning 0.45.0, jax/jaxlib 0.10.2, iqm-client 36.0.0, qiskit 2.1.2,
  qiskit-aer 0.17.2, numpy 2.5.4, scipy 1.18.1.
  The first install resolved jax 0.12.0.1; jax.__version__ reports that string.
  PennyLane 0.45.1 fails at runtime with jax 0.12.0.1 and 0.11.2:
    AttributeError: module 'jax.core' has no attribute 'is_concrete'
  It passes the jit+grad+vmap smoke test with 0.10.2 and 0.9.2. Pinned to 0.10.2.

Phase 2 labels
  Ran: qml/run_qml.py --only labels (qml/labels.py at 0d8bef1; on AC).
  Design: one optimize_sweep (preset lim), 15 n_X x 4125 grid points = 61875 points.
  Sweep wall time 275.9 s.
  Output: qml/data/labels_0d8bef1402_9fa16da52f39.{npz,json}, SHA256SUMS.
  Quality gate vs the reference JSONs, at every shared point that is not cliff_edge:
    main: 103 points, 0 worse, min relative difference -9.5e-14
    tail: 175 points, 0 worse, min relative difference -9.5e-13
    PASS
  Label horizons minus reference horizons (km): 1e4 -0.0002, 1e5 -0.0009, 1e6 -0.0071, 1e7 -0.0057, 1e8 -0.0073,
  1e9 -0.0021 (max |diff| 0.0073 km; the label sweep's horizon bisection is warm-started, 30 steps).
  Scored points by split:
    train  bulk 3407, tail 1320
    val    bulk 3402, tail 1320
    test   bulk 6797, tail 2640
    ref    bulk 88,   tail 175
    heldout (3e6, 5e8)  bulk 5748, tail 960
    extrap  (5e3, 2e9)  bulk 3812, tail 912
  401 scored points fall in crossover windows (Amendment 2). noise_floor 6.

L = 0 overlap (labels split rule):
  The reference grids contain L = 0 and L = 200 km. Both lie on the label lattice.
  The 12 points (6 reference n_X x {0, 200} km) keep their lattice split (train) and are not scored as ref.
  This keeps PREREGISTRATION section 4's train rule. It contradicts that section's sentence that reference
  points are "not on the lattice".
  The L = 200 km points have R <= 0, so they are not trained on either.
  Effect: the ref split scores 88 bulk + 175 tail points, instead of 103 + 175 minus the 9 points shared by
  both reference files.

Below-horizon branch switches (objective R) logged by the dense label sweep: 13 (table in PREREGISTRATION
Amendment 2).
  At the report's six n_X:
    1e4: none
    1e5: 130.547 km
    1e6: 118.547 km
    1e7: 105.953 and 138.234 km
    1e8: 134.766 km
    1e9: 128.672 km
  The benchmark sweep (results/benchmark_lim_0c8f47a1f0_530f6d58460c) logged only n_X=1e5 and 1e6.
  Its crossovers 130.3-130.8 and 118.3-118.8 km contain the label sweep's 130.547 and 118.547.
  Effect on the report: final_report.tex Section 5.3 says the sweep "logged two switches between competing
  local optima while the key is positive" and "No switch was logged below the horizon for n_X = 10^4, 10^7,
  10^8 and 10^9".
    - That is accurate for the benchmark sweep.
    - The denser label sweep shows at least 4 further switches at n_X = 1e7 (2), 1e8 and 1e9. They have
      u-distance 0.005-0.014 and gain at detection 2.8e-5 to 1.2e-3.
  So the report's statement is incomplete for 1e7, 1e8 and 1e9. The report was NOT edited (as instructed).

cliff_edge certificate (finding; report not edited):
  At all 7 reference cliff_edge points:
    - label R and l equal the reference values;
    - the label grad_norm_theta is 3e-13 .. 1.4e-7, while the reference stores 9.7e3 .. 1.8e9.
  The registered rule (grad > 1e-3) therefore flags 0 label points.
  Re-certified with the current code (certify_batch, objective R, scale |R|), the reference's own parameters
  give 3e-13 .. 1.1e-6.
  Strongest evidence: n_X=1e4, L=130.326 km appears in both reference files.
    - Their parameters agree to |du|_inf 2.1e-11 and R agrees to 11 digits.
    - The stored gradients are 9.96e3 and 1.32e9.
  So the stored cliff_edge gradient is not a property of the solution.
  Hypothesis (not yet tested): the certificate's normalisation, scale = |value at the best start| (optimize._solve),
  inflates the gradient by about R/scale when the best start had a tiny key rate.
  Affected text:
    - final_report.tex Section 4.1 ("where the greatest fixed point of eps_sec = kappa*l disappears and R(p) is
      discontinuous")
    - STATUS section A group (i)
  Both may be unsupported. To be resolved outside this study.
  In this study:
    - label points: the registered rule as written (0 flagged);
    - ref split: the reference's own flags.

Tests: tests/qml/test_qml_main.py, 4 tests OK in .venv.
  tests/qml/test_qml_models.py: skipped in .venv with the reason printed; 2 tests OK in .venv-qml.
  The model test found PQC_GRID out of order relative to the parameter-count matching of PREREGISTRATION
  section 5. It was reordered to (5,3) (6,3) (8,3) (5,6) (6,6) (8,6) = 75 90 120 150 180 240 before any training.
