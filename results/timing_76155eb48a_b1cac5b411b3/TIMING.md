# Timing benchmark (measured on this machine)

git `76155eb48a`, config `b1cac5b411b3`, Apple M2 Pro (12 logical CPUs), JAX 0.9.0 `cpu`, float64. Legacy preset physics.

| method | quantity | median | p90 | notes |
|---|---|---|---|---|
| JAX warm-started solver | batch-1 steady-state, device-resident (µs) | 297 | 333 | 2 Newton iterations, warm start from L−0.5 km |
| JAX warm-started solver | end-to-end Python API (µs) | 578 | 685 | numpy in → numpy out |
| lookup-table interpolation | batch-1 steady-state, device-resident (µs) | 8.2 | 8.9 | table [6, 81, 5] |
| lookup-table interpolation | end-to-end (µs) | 10.7 | 10.9 | |
| legacy DA (archived script call) | one point (s) | 3.3 | – | L=110.0 km, n_X=1e+09, maxiter=5000, nfev=50005, 66 µs/eval |
| legacy DA (archived script call) | one point (s) | 6.7 | – | L=150.0 km, n_X=1e+09, maxiter=10000, nfev=100009, 67 µs/eval |
| legacy DA (archived script call) | one point (s) | 9.9 | – | L=190.0 km, n_X=1e+09, maxiter=15000, nfev=150013, 66 µs/eval |
| legacy DA (archived script call) | one point (s) | 5.0 | – | L=130.0 km, n_X=1e+04, maxiter=7500, nfev=75007, 67 µs/eval |
| legacy DA (archived script call) | one point (s) | 6.6 | – | L=145.0 km, n_X=1e+04, maxiter=10000, nfev=100009, 66 µs/eval |
| legacy L-BFGS-B branch (L ≤ 100 km) | one point (s) | 0.002 | – | L=0.0 km, n_X=1e+09, nfev=19 |
| legacy L-BFGS-B branch (L ≤ 100 km) | one point (s) | 0.001 | – | L=50.0 km, n_X=1e+09, nfev=10 |
| legacy L-BFGS-B branch (L ≤ 100 km) | one point (s) | 0.001 | – | L=100.0 km, n_X=1e+09, nfev=8 |
| legacy L-BFGS-B branch (L ≤ 100 km) | one point (s) | 0.000 | – | L=50.0 km, n_X=1e+04, nfev=5 |
| NN surrogate | – | – | – | not measured (torch not installed) |
