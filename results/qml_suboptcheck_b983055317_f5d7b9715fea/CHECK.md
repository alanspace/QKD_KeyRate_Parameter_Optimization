# Known sub-optimal labels vs cold solves (repaired: labels_repaired_b983055317_9dd20e4718fe.npz)

Checked 497 lattice points (stretches +-1.0 km). Originally sub-optimal (> 1e-09): 77. Of those failing after repair: 0. Any point failing after repair: 0.

| n_X | stretch km | source | expected | found | in stretch | max orig deficit | fail after repair | window pts | window fails |
|---|---|---|---|---|---|---|---|---|---|
| 1e+06 | 138.53-138.75 | STATUS 10d | 8 | 8 | 8 | 2.29e-04 | 0 | 72 | 0 |
| 3e+06 | 138.94-139.16 | STATUS 10d | 8 | 8 | 7 | 2.51e-04 | 0 | 71 | 0 |
| 3.16e+06 | 138.91-139.22 | STATUS 10d | 11 | 11 | 10 | 3.23e-04 | 0 | 74 | 0 |
| 3.16e+07 | 98.75-99.38 | STATUS 10d | 6 | 6 | 6 | 5.96e-04 | 0 | 31 | 0 |
| 3.16e+07 | 136.84-137.19 | STATUS 10d | 12 | 12 | 12 | 2.48e-04 | 0 | 76 | 0 |
| 1e+08 | 92.25-92.38 | STATUS 10d | 2 | 2 | 2 | 2.75e-04 | 0 | 18 | 0 |
| 3.16e+08 | 84.88-85.25 | STATUS 10d | 4 | 4 | 3 | 1.30e-04 | 0 | 19 | 0 |
| 5e+08 | 82.0-82.25 | STATUS 10d | 3 | 3 | 3 | 6.94e-05 | 0 | 19 | 0 |
| 1e+09 | 77.63-77.75 | STATUS 10d | 2 | 2 | 1 | 3.42e-05 | 0 | 17 | 0 |
| 2e+09 | 73.13-73.25 | STATUS 10d | 2 | 2 | 1 | 2.76e-05 | 0 | 17 | 0 |
| 1e+07 | 105.375-105.938 | STATUS 11(c) | 19 | 19 | 19 | 7.47e-04 | 0 | 83 | 0 |

Per point (originally sub-optimal, plus any failing point):

| n_X | L km | split | orig deficit | repaired vs cold | repaired flag | result |
|---|---|---|---|---|---|---|
| 1e+06 | 138.5312 | none | 6.24e-06 | 3.24e-15 | True | PASS |
| 1e+06 | 138.5625 | none | 3.78e-05 | 3.10e-15 | True | PASS |
| 1e+06 | 138.5938 | none | 6.95e-05 | -4.74e-15 | True | PASS |
| 1e+06 | 138.6250 | val | 1.01e-04 | -1.63e-15 | True | PASS |
| 1e+06 | 138.6562 | none | 1.33e-04 | 1.49e-15 | True | PASS |
| 1e+06 | 138.6875 | none | 1.65e-04 | 4.77e-15 | True | PASS |
| 1e+06 | 138.7188 | none | 1.97e-04 | 1.64e-15 | True | PASS |
| 1e+06 | 138.7500 | test | 2.29e-04 | 7.19e-15 | True | PASS |
| 3e+06 | 138.9375 | heldout | 2.58e-05 | 3.87e-15 | True | PASS |
| 3e+06 | 138.9688 | heldout | 5.77e-05 | -3.88e-15 | True | PASS |
| 3e+06 | 139.0000 | heldout | 8.97e-05 | 2.05e-15 | True | PASS |
| 3e+06 | 139.0312 | heldout | 1.22e-04 | -1.85e-15 | True | PASS |
| 3e+06 | 139.0625 | heldout | 1.54e-04 | -2.47e-15 | True | PASS |
| 3e+06 | 139.0938 | heldout | 1.86e-04 | 2.88e-15 | True | PASS |
| 3e+06 | 139.1250 | heldout | 2.19e-04 | -4.13e-16 | True | PASS |
| 3e+06 | 139.1562 | heldout | 2.51e-04 | 1.45e-15 | True | PASS |
| 3.16e+06 | 138.9062 | none | 1.71e-06 | 3.60e-15 | True | PASS |
| 3.16e+06 | 138.9375 | none | 3.35e-05 | -1.60e-15 | True | PASS |
| 3.16e+06 | 138.9688 | none | 6.54e-05 | 1.21e-15 | True | PASS |
| 3.16e+06 | 139.0000 | train | 9.73e-05 | 1.01e-15 | True | PASS |
| 3.16e+06 | 139.0312 | none | 1.29e-04 | 3.43e-15 | True | PASS |
| 3.16e+06 | 139.0625 | none | 1.61e-04 | 1.62e-15 | True | PASS |
| 3.16e+06 | 139.0938 | none | 1.94e-04 | 1.62e-15 | True | PASS |
| 3.16e+06 | 139.1250 | val | 2.26e-04 | 0.00e+00 | True | PASS |
| 3.16e+06 | 139.1562 | none | 2.58e-04 | 1.63e-15 | True | PASS |
| 3.16e+06 | 139.1875 | none | 2.91e-04 | 1.63e-15 | True | PASS |
| 3.16e+06 | 139.2188 | none | 3.23e-04 | 3.88e-15 | True | PASS |
| 3.16e+07 | 98.7500 | test | 8.30e-05 | -2.43e-15 | True | PASS |
| 3.16e+07 | 98.8750 | test | 1.84e-04 | 2.23e-16 | True | PASS |
| 3.16e+07 | 99.0000 | train | 2.86e-04 | 1.12e-16 | True | PASS |
| 3.16e+07 | 99.1250 | val | 3.89e-04 | -1.13e-16 | True | PASS |
| 3.16e+07 | 99.2500 | test | 4.92e-04 | 1.81e-15 | True | PASS |
| 3.16e+07 | 99.3750 | test | 5.96e-04 | 2.05e-15 | True | PASS |
| 3.16e+07 | 136.8438 | none | 6.92e-06 | 2.25e-15 | True | PASS |
| 3.16e+07 | 136.8750 | test | 2.86e-05 | 1.03e-15 | True | PASS |
| 3.16e+07 | 136.9062 | none | 5.03e-05 | -4.11e-16 | True | PASS |
| 3.16e+07 | 136.9375 | none | 7.20e-05 | 1.65e-15 | True | PASS |
| 3.16e+07 | 136.9688 | none | 9.38e-05 | 8.26e-16 | True | PASS |
| 3.16e+07 | 137.0000 | train | 1.16e-04 | 1.24e-15 | True | PASS |
| 3.16e+07 | 137.0312 | none | 1.38e-04 | -3.94e-15 | True | PASS |
| 3.16e+07 | 137.0625 | none | 1.59e-04 | -3.11e-15 | True | PASS |
| 3.16e+07 | 137.0938 | none | 1.81e-04 | -6.24e-16 | True | PASS |
| 3.16e+07 | 137.1250 | val | 2.03e-04 | 2.29e-15 | True | PASS |
| 3.16e+07 | 137.1562 | none | 2.26e-04 | 1.67e-15 | True | PASS |
| 3.16e+07 | 137.1875 | none | 2.48e-04 | 4.39e-15 | True | PASS |
| 1e+08 | 92.2500 | test | 2.10e-04 | -8.31e-16 | True | PASS |
| 1e+08 | 92.3750 | test | 2.75e-04 | 1.67e-15 | True | PASS |
| 3.16e+08 | 84.8750 | test | 9.96e-06 | -6.89e-16 | True | PASS |
| 3.16e+08 | 85.0000 | train | 4.97e-05 | -3.47e-16 | True | PASS |
| 3.16e+08 | 85.1250 | val | 8.97e-05 | 2.09e-15 | True | PASS |
| 3.16e+08 | 85.2500 | test | 1.30e-04 | 1.05e-15 | True | PASS |
| 5e+08 | 82.0000 | heldout | 3.70e-06 | 1.44e-15 | True | PASS |
| 5e+08 | 82.1250 | heldout | 3.64e-05 | 1.74e-15 | True | PASS |
| 5e+08 | 82.2500 | heldout | 6.94e-05 | -1.75e-15 | True | PASS |
| 1e+09 | 77.6250 | val | 9.79e-06 | 0.00e+00 | True | PASS |
| 1e+09 | 77.7500 | test | 3.42e-05 | -2.55e-15 | True | PASS |
| 2e+09 | 73.1250 | extrap | 9.42e-06 | 1.36e-15 | True | PASS |
| 2e+09 | 73.2500 | extrap | 2.76e-05 | 1.19e-15 | True | PASS |
| 1e+07 | 105.3750 | test | 2.85e-05 | -1.12e-15 | True | PASS |
| 1e+07 | 105.4062 | none | 6.78e-05 | 1.87e-15 | True | PASS |
| 1e+07 | 105.4375 | none | 1.07e-04 | 1.87e-16 | True | PASS |
| 1e+07 | 105.4688 | none | 1.47e-04 | -1.12e-15 | True | PASS |
| 1e+07 | 105.5000 | train | 1.86e-04 | 1.12e-15 | True | PASS |
| 1e+07 | 105.5312 | none | 2.26e-04 | 3.76e-16 | True | PASS |
| 1e+07 | 105.5625 | none | 2.66e-04 | -1.32e-15 | True | PASS |
| 1e+07 | 105.5938 | none | 3.05e-04 | 3.96e-15 | True | PASS |
| 1e+07 | 105.6250 | val | 3.45e-04 | 1.70e-15 | True | PASS |
| 1e+07 | 105.6562 | none | 3.85e-04 | -1.89e-15 | True | PASS |
| 1e+07 | 105.6875 | none | 4.25e-04 | -2.27e-15 | True | PASS |
| 1e+07 | 105.7188 | none | 4.65e-04 | 1.90e-16 | True | PASS |
| 1e+07 | 105.7500 | test | 5.05e-04 | 4.75e-15 | True | PASS |
| 1e+07 | 105.7812 | none | 5.45e-04 | 2.66e-15 | True | PASS |
| 1e+07 | 105.8125 | none | 5.85e-04 | 4.38e-15 | True | PASS |
| 1e+07 | 105.8438 | none | 6.26e-04 | 3.81e-16 | True | PASS |
| 1e+07 | 105.8750 | test | 6.66e-04 | 1.91e-16 | True | PASS |
| 1e+07 | 105.9062 | none | 7.06e-04 | -5.74e-16 | True | PASS |
| 1e+07 | 105.9375 | none | 7.47e-04 | -1.92e-15 | True | PASS |
