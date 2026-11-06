# Independent solver cross-check — preset `lim`

`Optimization/reference/crosscheck_scipy.py`; git `d605664fc3`, config `57b813e01a63`; reference `Optimization/reference/reference_lim.json` (config `5266f6d5631d`).

d = (R_ref − R_xcheck)/R_ref; d > 0: cross-check worse than reference; d < −1e-9: REFERENCE WEAKNESS.

- points (R_ref > 0): 105
- |d| ≤ 1e-6: 98.1 %
- median d: -7.87e-14; max d: 1.00e+00; min d: -5.12e-10
- cross-check worse than reference by > 1e-6: 2 points
- **reference weaknesses (cross-check better by > 1e-9): 0**
- SLSQP runs not reporting success: 3 of 1680; infeasible results: 3; non-finite: 0; points hitting the 300 s budget: 0; points with no feasible result: 0
