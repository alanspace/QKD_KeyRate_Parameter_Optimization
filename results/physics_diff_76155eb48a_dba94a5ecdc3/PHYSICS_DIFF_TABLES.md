# Physics differences and paper targets (generated)

`Optimization/analysis/physics_diff.py`, git `76155eb48a`, config `dba94a5ecdc3`, physics hash `e6edcf3d1d1d`.

## Cumulative chain legacy → lim_exact (optimised R, bits/pulse)

| stage | L=0, n_X=1e+04 | L=50, n_X=1e+04 | L=100, n_X=1e+04 | L=150, n_X=1e+04 | L=0, n_X=1e+09 | L=50, n_X=1e+09 | L=100, n_X=1e+09 | L=150, n_X=1e+09 |
|---|---|---|---|---|---|---|---|---|
| 0 legacy (src/qkd switches) | 3.800e-04 | 3.543e-05 | 2.598e-06 | (-1.0e-25, l≤0) | 1.090e-02 | 1.052e-03 | 9.881e-05 | 5.855e-06 |
| 1 m_Z = total Z errors (Eq. 4) | 8.386e-04 | 7.770e-05 | 5.393e-06 | (-7.2e-27, l≤0) | 1.567e-02 | 1.522e-03 | 1.420e-04 | 1.006e-05 |
| 2 e_mis 0.01 -> 5e-3 | 1.233e-03 | 1.145e-04 | 8.199e-06 | 1.079e-09 | 1.911e-02 | 1.860e-03 | 1.736e-04 | 1.238e-05 |
| 3 p_ap 0 -> 0.04 in e_k | 2.812e-04 | 2.530e-05 | 1.221e-06 | (-4.5e-25, l≤0) | 8.966e-03 | 8.631e-04 | 7.894e-05 | 5.055e-06 |
| 4 R_k = D_k(1+p_ap) in counts/N | 3.148e-04 | 2.847e-05 | 1.471e-06 | (-3.4e-25, l≤0) | 9.670e-03 | 9.318e-04 | 8.565e-05 | 5.646e-06 |
| 5 eps_sec = kappa*l (kappa=1e-15) | 2.651e-04 | 2.374e-05 | 1.067e-06 | (-9.3e-25, l≤0) | 9.930e-03 | 9.570e-04 | 8.801e-05 | 5.861e-06 |
| 6 no clamp of n^-, m^- at 0 | 2.084e-04 | 1.883e-05 | 9.913e-07 | (-9.3e-25, l≤0) | 9.203e-03 | 8.897e-04 | 8.537e-05 | 5.861e-06 |
| 7 eta_ch (paper text) in e_mis term = lim_exact | 4.897e-08 | (-6.0e-25, l≤0) | (-2.5e-24, l≤0) | (-2.9e-08, l≤0) | 1.250e-03 | 9.616e-05 | 7.939e-06 | (-7.8e-27, l≤0) |

## Paper targets (15 % tolerance, never tuned)

| model | T1 R(0,1e9) [~1e-2] | T2 ratio [~1.75] | T3 L at R=1e-7, n_X=1e4 [~135 km] | horizon n_X=1e4 (km) |
|---|---|---|---|---|
| lim_exact | 1.2505e-03 (MISMATCH) | 2.734 (MISMATCH) | R < 1e-7 on the whole 60-180 km grid (MISMATCH) | None |
| lim_exact[emis_eta=sys] (diagnostic, NOT a preset) | 9.2025e-03 (ok) | 1.791 (ok) | 118.78 (ok) | 130.4409484853968 |
| legacy | 1.0900e-02 (ok) | 1.790 (ok) | 134.14 (ok) | 148.17417520377785 |
