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
