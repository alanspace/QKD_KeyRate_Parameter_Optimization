# Independent solver cross-check — preset `legacy`

`Optimization/reference/crosscheck_scipy.py`; git `d605664fc3`, config `7fd2df0797b2`; reference `Optimization/reference/reference_legacy.json` (config `57bd0e7de31c`).

d = (R_ref − R_xcheck)/R_ref; d > 0: cross-check worse than reference; d < −1e-9: REFERENCE WEAKNESS.

- points (R_ref > 0): 110
- |d| ≤ 1e-6: 97.3 %
- median d: -8.60e-14; max d: 1.00e+00; min d: -1.16e-10
- cross-check worse than reference by > 1e-6: 3 points
- **reference weaknesses (cross-check better by > 1e-9): 0**
- SLSQP runs not reporting success: 24 of 1760; infeasible results: 25; non-finite: 0; points hitting the 300 s budget: 0; points with no feasible result: 0
