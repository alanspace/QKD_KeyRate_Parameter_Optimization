# Phase 3 timing pilot (STOP 2)

Inputs (measured, on AC, Apple M2 Pro CPU):
- `results/qml_pilot_train_d70559dfbe_e35551f03c57/pilot_train.json` (.venv-qml, jax 0.10.2, pennylane 0.45.1)
- `results/qml_pilot_certify_d118356a45_b38f98f8dcfe/pilot_certify.json` (.venv, jax 0.9.0)

## Model side: one full-batch Adam step on 4727 training points (steady median of 3), and prediction of the 45500-point query set

| family | config | params | step s | predict s |
|---|---|---|---|---|
| PQC-full | [5, 3] | 75 | 0.024 | 0.107 |
| PQC-full | [6, 3] | 90 | 0.052 | 0.249 |
| PQC-full | [8, 3] | 120 | 0.296 | 1.401 |
| PQC-full | [5, 6] | 150 | 0.047 | 0.220 |
| PQC-full | [6, 6] | 180 | 0.128 | 0.486 |
| PQC-full | [8, 6] | 240 | 0.598 | 2.633 |
| PQC-product | [5, 3] | 75 | 0.019 | 0.093 |
| PQC-product | [6, 3] | 90 | 0.042 | 0.208 |
| PQC-product | [8, 3] | 120 | 0.258 | 1.100 |
| PQC-product | [5, 6] | 150 | 0.037 | 0.177 |
| PQC-product | [6, 6] | 180 | 0.109 | 0.412 |
| PQC-product | [8, 6] | 240 | 0.517 | 2.155 |
| NN | 5 | 75 | 0.000 | 0.001 |
| NN | 6 | 95 | 0.001 | 0.001 |
| NN | 7 | 117 | 0.001 | 0.001 |
| NN | 8 | 141 | 0.001 | 0.002 |
| NN | 10 | 195 | 0.001 | 0.002 |
| NN | 12 | 257 | 0.001 | 0.002 |

- RFF ridge: fit with a 5-value lambda sweep takes 0.002-0.004 s per configuration.
- Lookup tables: build and predict take 0.004-0.009 s.

## Physics side: certification and hybrid

- Certification of 1000 predictions (R, l, regret): 0.005 s steady (0.5 s first call).
- Hybrid k=1, batched over 1000 points: 0.097 s; stand-in median regret 2.00e-05 (median 4.07e-03 before Newton; stand-in = label u + N(0, 0.01)).
- Hybrid k=2, batched over 1000 points: 0.177 s; stand-in median regret 2.04e-09 (median 4.07e-03 before Newton; stand-in = label u + N(0, 0.01)).
- Hybrid k=3, batched over 1000 points: 0.257 s; stand-in median regret 2.84e-15 (median 4.07e-03 before Newton; stand-in = label u + N(0, 0.01)).

Batch-1 timings:

| quantity | device median (µs) | end-to-end median (µs) |
|---|---|---|
| Newton k=1 (n=50) | 685 | 896 |
| Newton k=2 (n=50) | 1146 | 1368 |
| Newton k=3 (n=50) | 1608 | 1835 |
| warm solve from nearest lower training node (median 0.25 km, 2 iterations; n=50) | - | 1629 |
| cold solve, optimize(), 2^12 Sobol, top-8 (n=20; first measurement) | - | 29711 (p90 32771) |

## Projection (E = 2000 steps; 6 configs x 3 trained families for tuning; 5 seeds for the selected configs)

| part | seconds |
|---|---|
| tuning (18 trained configurations, including compiles) | 4295 |
| final dense training, 5 seeds (PQC-full, PQC-product, NN-tuned, NN-matched) | 435 (smallest configs selected) to 11169 (largest) |
| sparse variant (Amendment 3; 940 training points), 5 seeds | 86 to 2221 (step time scaled linearly with training-set size: an estimate) |
| validation certification (24 configs incl. ridge) | 1 |
| test certification + hybrid k=1..3 (27 prediction sets x 21132 eval points + 45500 query) | 309 |
| predictions | 81 |
| scenario (i) cold baseline, batched 40 ms/point over 21132 points | 845 |
| scenario (ii) warm baseline | 7 |
| **total** | **1.68 h (best) to 5.26 h (worst), including the sparse variant; 4.64 h worst without it** |

Below 12 h: the sparse variant (Amendment 3) is included.
