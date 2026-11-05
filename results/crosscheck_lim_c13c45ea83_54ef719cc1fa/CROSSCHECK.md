# Independent solver cross-check — preset `lim`

`Optimization/reference/crosscheck_scipy.py`; git `c13c45ea83`, config `54ef719cc1fa`; reference `Optimization/reference/reference_lim.json` (config `4aca6b195b78`).

d = (R_ref − R_xcheck)/R_ref; d > 0: cross-check worse than reference; d < −1e-9: REFERENCE WEAKNESS.

- points (R_ref > 0): 105
- |d| ≤ 1e-6: 98.1 %
- median d: -7.47e-14; max d: 1.00e+00; min d: -3.55e-10
- cross-check worse than reference by > 1e-6: 2 points
- **reference weaknesses (cross-check better by > 1e-9): 0**
- SLSQP runs not reporting success: 14 of 1680; infeasible results: 14; non-finite: 0; points hitting the 300 s budget: 0; points with no feasible result: 0
