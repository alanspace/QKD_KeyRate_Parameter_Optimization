# Independent solver cross-check — preset `lim`

`Optimization/reference/crosscheck_scipy.py`; git `9b2938e502`, config `5f2044220b48`; reference `Optimization/reference/reference_lim.json` (config `5671a043fc90`).

d = (R_ref − R_xcheck)/R_ref; d > 0: cross-check worse than reference; d < −tol_ref: REFERENCE WEAKNESS.

- points (R_ref > 0): 105
- |d| ≤ 1e-6: 98.1 %
- median d: 1.90e-15; max d: 1.00e+00; min d: -2.66e-15
- cross-check worse than reference by > 1e-6: 2 points
- tol_ref = max(1e-9, 2 x seed disagreement of the reference) = 1.000e-09
- **reference weaknesses (cross-check better by > tol_ref): 0**
- INFO (cross-check better by 1e-9 .. tol_ref): 0
- SLSQP runs not reporting success: 11 of 1680; infeasible results: 12; non-finite: 0; points hitting the 300 s budget: 0; points with no feasible result: 0
