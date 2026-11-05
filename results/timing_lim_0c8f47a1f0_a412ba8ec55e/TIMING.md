# Timing benchmark (measured on this machine)

git `0c8f47a1f0`, config `a412ba8ec55e`, Apple M2 Pro (12 logical CPUs), JAX 0.9.0 `cpu`, float64. Preset `lim`.

| method | quantity | median | p90 | notes |
|---|---|---|---|---|
| JAX warm-started solver | batch-1 steady-state, device-resident (µs) | 1199 | 1322 | 2 Newton iterations, warm start from L−0.5 km |
| JAX warm-started solver | end-to-end Python API (µs) | 1466 | 1555 | numpy in → numpy out |
| JAX warm-started solver, tail (horizon 130.44 km) | batch-1 steady-state device / end-to-end (µs) | 2123 / 2460 | 2244 / 2610 | L=125.441 km, n_X=1e+04, 4 Newton iterations |
| JAX warm-started solver, tail (horizon 160.61 km) | batch-1 steady-state device / end-to-end (µs) | 2098 / 2490 | 2193 / 2579 | L=155.606 km, n_X=1e+05, 4 Newton iterations |
| JAX warm-started solver, tail (horizon 172.62 km) | batch-1 steady-state device / end-to-end (µs) | 2096 / 2438 | 2199 / 2573 | L=167.624 km, n_X=1e+06, 4 Newton iterations |
| JAX warm-started solver, tail (horizon 178.67 km) | batch-1 steady-state device / end-to-end (µs) | 2554 / 2903 | 2640 / 2988 | L=173.668 km, n_X=1e+07, 5 Newton iterations |
| JAX warm-started solver, tail (horizon 182.09 km) | batch-1 steady-state device / end-to-end (µs) | 2094 / 2481 | 2172 / 2664 | L=177.091 km, n_X=1e+08, 4 Newton iterations |
| JAX warm-started solver, tail (horizon 184.16 km) | batch-1 steady-state device / end-to-end (µs) | 1654 / 2024 | 1732 / 2114 | L=179.16 km, n_X=1e+09, 3 Newton iterations |
| lookup-table interpolation | batch-1 steady-state, device-resident (µs) | 7.7 | 8.4 | table [6, 81, 5] |
| lookup-table interpolation | end-to-end (µs) | 10.0 | 10.5 | |
| legacy DA (archived settings, this preset) | one point (s) | 5.8 | – | L=110.0 km, n_X=1e+09, maxiter=5000, nfev=50005, 116 µs/eval |
| legacy DA (archived settings, this preset) | one point (s) | 11.7 | – | L=150.0 km, n_X=1e+09, maxiter=10000, nfev=100009, 117 µs/eval |
| legacy DA (archived settings, this preset) | one point (s) | 17.6 | – | L=190.0 km, n_X=1e+09, maxiter=15000, nfev=150013, 117 µs/eval |
| legacy DA (archived settings, this preset) | one point (s) | 8.8 | – | L=130.0 km, n_X=1e+04, maxiter=7500, nfev=75007, 118 µs/eval |
| legacy DA (archived settings, this preset) | one point (s) | 11.7 | – | L=145.0 km, n_X=1e+04, maxiter=10000, nfev=100009, 117 µs/eval |
| legacy L-BFGS-B branch (L ≤ 100 km) | one point (s) | 0.010 | – | L=0.0 km, n_X=1e+09, nfev=24 |
| legacy L-BFGS-B branch (L ≤ 100 km) | one point (s) | 0.003 | – | L=50.0 km, n_X=1e+09, nfev=21 |
| legacy L-BFGS-B branch (L ≤ 100 km) | one point (s) | 0.004 | – | L=100.0 km, n_X=1e+09, nfev=29 |
| legacy L-BFGS-B branch (L ≤ 100 km) | one point (s) | 0.001 | – | L=50.0 km, n_X=1e+04, nfev=5 |
| NN surrogate | – | – | – | not measured (torch not installed) |
