# Physics differences and paper targets (generated)

`Optimization/analysis/physics_diff.py`, git `a419cc7422`, config `dba94a5ecdc3`, physics hash `9564691bc5b0`.

## Cumulative chain legacy → lim → lim_literal (optimised R, bits/pulse)

| stage | L=0, n_X=1e+04 | L=50, n_X=1e+04 | L=100, n_X=1e+04 | L=150, n_X=1e+04 | L=0, n_X=1e+09 | L=50, n_X=1e+09 | L=100, n_X=1e+09 | L=150, n_X=1e+09 |
|---|---|---|---|---|---|---|---|---|
| 0 legacy (src/qkd switches) | 3.800e-04 | 3.543e-05 | 2.598e-06 | (-1.0e-25, l≤0) | 1.090e-02 | 1.052e-03 | 9.881e-05 | 5.855e-06 |
| 1 expm1 evaluation of 1-exp(-eta k) (numerics only) | 3.800e-04 | 3.543e-05 | 2.598e-06 | (-1.0e-25, l≤0) | 1.090e-02 | 1.052e-03 | 9.881e-05 | 5.855e-06 |
| 2 m_Z = total Z errors (Eq. 4) | 8.386e-04 | 7.770e-05 | 5.393e-06 | (-7.2e-27, l≤0) | 1.567e-02 | 1.522e-03 | 1.420e-04 | 1.006e-05 |
| 3 e_mis 0.01 -> 5e-3 | 1.233e-03 | 1.145e-04 | 8.199e-06 | 1.079e-09 | 1.911e-02 | 1.860e-03 | 1.736e-04 | 1.238e-05 |
| 4 p_ap 0 -> 0.04 in e_k | 2.812e-04 | 2.530e-05 | 1.221e-06 | (-4.5e-25, l≤0) | 8.966e-03 | 8.631e-04 | 7.894e-05 | 5.055e-06 |
| 5 R_k = D_k(1+p_ap) in counts/N | 3.148e-04 | 2.847e-05 | 1.471e-06 | (-3.4e-25, l≤0) | 9.670e-03 | 9.318e-04 | 8.565e-05 | 5.646e-06 |
| 6 eps_sec = kappa*l (kappa=1e-15) | 2.651e-04 | 2.374e-05 | 1.067e-06 | (-9.3e-25, l≤0) | 9.930e-03 | 9.570e-04 | 8.801e-05 | 5.861e-06 |
| 7 no clamp of n^-, m^- at 0 = lim | 2.084e-04 | 1.883e-05 | 9.913e-07 | (-9.3e-25, l≤0) | 9.203e-03 | 8.897e-04 | 8.537e-05 | 5.861e-06 |
| 8 eta_ch (as printed) in e_mis term = lim_literal | 4.897e-08 | (-6.0e-25, l≤0) | (-2.5e-24, l≤0) | (-2.9e-08, l≤0) | 1.250e-03 | 9.616e-05 | 7.939e-06 | (-7.8e-27, l≤0) |

## Chain horizons (km, largest L with l > 0)

| stage | n_X=1e4 | n_X=1e9 |
|---|---|---|
| 0 legacy (src/qkd switches) | 148.17416429519653 | 190.6661593914032 |
| 1 expm1 evaluation of 1-exp(-eta k) (numerics only) | 148.17416429519653 | 190.6661593914032 |
| 2 m_Z = total Z errors (Eq. 4) | 148.17416429519653 | 190.6661593914032 |
| 3 e_mis 0.01 -> 5e-3 | 152.49786972999573 | 192.7118682861328 |
| 4 p_ap 0 -> 0.04 in e_k | 133.61459970474243 | 182.43325233459473 |
| 5 R_k = D_k(1+p_ap) in counts/N | 135.37303686141968 | 184.17141914367676 |
| 6 eps_sec = kappa*l (kappa=1e-15) | 130.44356107711792 | 184.16235327720642 |
| 7 no clamp of n^-, m^- at 0 = lim | 130.44356107711792 | 184.16235327720642 |
| 8 eta_ch (as printed) in e_mis term = lim_literal | 18.97205352783203 | 149.3087613582611 |

## Paper targets (15 % tolerance, never tuned)

| model | T1 R(0,1e9) [~1e-2] | T2 ratio [~1.75] | T3 horizon n_X=1e4 [~135 km] | T3b L at R=1e-7, n_X=1e4 (secondary) |
|---|---|---|---|---|
| lim | 9.2025e-03 (ok) | 1.791 (ok) | 130.44 (ok) | 118.78 (ok) |
| lim_literal | 1.2505e-03 (MISMATCH) | 2.734 (MISMATCH) | 18.97 (MISMATCH) | R < 1e-7 on the whole 60-180 km grid (MISMATCH) |
| legacy | 1.0900e-02 (ok) | 1.790 (ok) | 148.17 (ok) | 134.14 (ok) |
