# Independent solver cross-check — preset `lim` (tail reference)

`Optimization/reference/crosscheck_scipy.py`; git `9b2938e502`, config `ae76974bf9d9`; reference `Optimization/reference/reference_lim_tail.json` (config `f72424b20c69`).

d = (R_ref − R_xcheck)/R_ref; d > 0: cross-check worse than reference; d < −tol_ref: REFERENCE WEAKNESS.

- points (R_ref > 0): 180
- |d| ≤ 1e-6: 87.8 %
- median d: 7.74e-15; max d: 1.00e+00; min d: -2.01e-14
- cross-check worse than reference by > 1e-6: 22 points
- tol_ref = max(1e-9, 2 x seed disagreement of the reference) = 1.000e-09
- **reference weaknesses (cross-check better by > tol_ref): 0**
- INFO (cross-check better by 1e-9 .. tol_ref): 0
- SLSQP runs not reporting success: 167 of 2880; infeasible results: 236; non-finite: 0; points hitting the 300 s budget: 0; points with no feasible result: 0
