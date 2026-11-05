# Independent solver cross-check — preset `lim` (tail reference)

`Optimization/reference/crosscheck_scipy.py`; git `c13c45ea83`, config `0c31af16a8f7`; reference `Optimization/reference/reference_lim_tail.json` (config `1461dc9c8f26`).

d = (R_ref − R_xcheck)/R_ref; d > 0: cross-check worse than reference; d < −1e-9: REFERENCE WEAKNESS.

- points (R_ref > 0): 180
- |d| ≤ 1e-6: 88.3 %
- median d: -2.35e-11; max d: 1.10e+00; min d: -3.83e-09
- cross-check worse than reference by > 1e-6: 21 points
- **reference weaknesses (cross-check better by > 1e-9): 5**
- SLSQP runs not reporting success: 174 of 2880; infeasible results: 228; non-finite: 0; points hitting the 300 s budget: 0; points with no feasible result: 0
  - L=171.930 km, n_X=1e+06: R_ref=3.6889749357e-10, R_xcheck=3.6889749498e-10, d=-3.83e-09
  - L=178.947 km, n_X=1e+08: R_ref=3.3742089995e-08, R_xcheck=3.3742090030e-08, d=-1.04e-09
  - L=179.449 km, n_X=1e+08: R_ref=2.2438548469e-08, R_xcheck=2.2438548497e-08, d=-1.28e-09
  - L=181.454 km, n_X=1e+09: R_ref=3.6748169050e-08, R_xcheck=3.6748169107e-08, d=-1.55e-09
  - L=181.955 km, n_X=1e+09: R_ref=2.2967104125e-08, R_xcheck=2.2967104167e-08, d=-1.81e-09
