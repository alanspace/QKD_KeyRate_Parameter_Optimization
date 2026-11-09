# QML surrogate study: results (Phase 4)

Branch `qml-surrogate`. Protocol: Section 7 of `result_report/final_report.tex` and `qml/PREREGISTRATION.md`
(Amendments 1–5). Every number below is read from the result files listed at the end. Every regret is computed
from R re-evaluated by the float64 `lim` model at the predicted parameters. No model-predicted R is used anywhere.

**No quantum advantage is claimed.** The circuits are simulated classically, and the comparison is against classical
models of matched size.

## Summary

- **Labels.** All tables use the **repaired labels** (Amendment 4; `labels_repaired_b983055317_9dd20e4718fe.npz`), unless a table
  says otherwise. The certification against the original labels is reported once, as a secondary table.
- **Kill criterion** (dense PQC-full median bulk test regret above 1e-2, mean over seeds): **not met**.
  The value is 6.00e-03 ± 2.72e-03 (mean ± sd over 5 seeds; repaired labels). Against the original labels
  it is 6.00e-03 ± 2.72e-03.
- The usefulness verdicts per model and scenario are in the usefulness section below. Its definitions are post hoc
  and stated there, with the raw numbers next to them.
- **NN-tuned and NN-matched are the same configuration in this run** (both h = 12), so they are one result. Only NN-tuned is shown.
- **Cold-solve limit (STATUS 12).** At 21 label points a warm sweep beats the 2^14-Sobol/top-16 cold solve, by up to
  3.7e-4. The labels there are the best known optima, not proven global optima. Where these points enter a reported
  number is listed in the section "Points where the cold solve is beaten".

## Setup actually run

| item | value |
|---|---|
| labels | `15` block sizes, 61875 points; quality gate against the reference: True |
| label repair (Amendment 4) | 2914 labels replaced by the backward sweep; gate 4746 cold solves, 0 repaired labels worse; gate_pass True; crossovers 27 (original 13) |
| training points | dense 4727, sparse 940 |
| tuning | E = 2000 full-batch Adam steps, seed 0, 6 configurations per family |
| final runs | seeds [0, 1, 2, 3, 4], variants ['dense', 'sparse'] |
| evaluation points | 21137 |

## Hyperparameter selection (validation split; median regret, bulk and tail pooled, crossover windows excluded)

| family | config index | config | params | val median regret | val ≤1e-6 % | selected |
|---|---|---|---|---|---|---|
| NN | 0 | 5 | 75 | 1.33e-02 | 0.0 |  |
| NN | 1 | 6 | 95 | 2.61e-03 | 0.0 |  |
| NN | 2 | 7 | 117 | 3.03e-03 | 0.0 |  |
| NN | 3 | 8 | 141 | 1.03e-03 | 0.0 |  |
| NN | 4 | 10 | 195 | 2.50e-03 | 0.0 |  |
| NN | 5 | 12 | 257 | 9.08e-04 | 0.1 | **yes** |
| PQC-full | 0 | [5, 3] | 75 | 6.53e-02 | 0.0 |  |
| PQC-full | 1 | [6, 3] | 90 | 4.18e-02 | 0.0 |  |
| PQC-full | 2 | [8, 3] | 120 | 5.11e-02 | 0.0 |  |
| PQC-full | 3 | [5, 6] | 150 | 1.43e-02 | 0.0 |  |
| PQC-full | 4 | [6, 6] | 180 | 1.50e-02 | 0.0 |  |
| PQC-full | 5 | [8, 6] | 240 | 5.39e-03 | 0.0 | **yes** |
| PQC-product | 0 | [5, 3] | 75 | 4.83e-02 | 0.0 |  |
| PQC-product | 1 | [6, 3] | 90 | 2.35e-02 | 0.0 |  |
| PQC-product | 2 | [8, 3] | 120 | 4.36e-02 | 0.0 |  |
| PQC-product | 3 | [5, 6] | 150 | 1.01e-01 | 0.0 |  |
| PQC-product | 4 | [6, 6] | 180 | 1.99e-02 | 0.0 | **yes** |
| PQC-product | 5 | [8, 6] | 240 | 2.46e-02 | 0.0 |  |
| RFF-ridge | 0 | [5, 3] | 75 | 1.19e-01 | 0.0 |  |
| RFF-ridge | 1 | [6, 3] | 85 | 6.04e-02 | 0.0 |  |
| RFF-ridge | 2 | [8, 3] | 125 | 3.49e-01 | 0.0 |  |
| RFF-ridge | 3 | [5, 6] | 145 | 1.32e-02 | 0.0 | **yes** |
| RFF-ridge | 4 | [6, 6] | 185 | 1.47e-02 | 0.0 |  |
| RFF-ridge | 5 | [8, 6] | 245 | 6.73e-02 | 0.0 |  |

The NN-matched model uses the NN configuration with the same index as the selected PQC-full configuration.

## Primary metric: test splits, dense training (mean ± sd over seeds; lookups are deterministic)

| model | split | n | median | p99 | max | ≤1e-6 % | ≤1e-4 % | ≤1e-3 % | ≤1e-2 % | no key % | u-err median |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PQC-full | test_bulk | 6642 | 6.00e-03 ± 2.72e-03 | 5.51e-01 ± 4.93e-01 | 7.49e-01 ± 7.27e-01 | 0.0 ± 0.0 | 0.8 ± 0.9 | 14.9 ± 8.9 | 63.9 ± 9.7 | 0.2 ± 0.5 | 1.83e-02 ± 4.39e-03 |
| PQC-full | test_tail | 2608 | 4.67e-02 ± 1.22e-02 | 1.11e+03 ± 5.13e+02 | 3.11e+05 ± 9.07e+04 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.6 ± 0.5 | 15.1 ± 7.8 | 14.4 ± 2.6 | 4.16e-02 ± 9.89e-03 |
| PQC-product | test_bulk | 6642 | 1.43e-02 ± 8.10e-03 | 1.09e+00 ± 9.84e-01 | 1.62e+00 ± 1.04e+00 | 0.0 ± 0.0 | 0.1 ± 0.2 | 5.9 ± 2.4 | 43.9 ± 11.9 | 2.1 ± 3.5 | 2.38e-02 ± 5.41e-03 |
| PQC-product | test_tail | 2608 | 5.95e-02 ± 1.85e-02 | 1.54e+03 ± 8.04e+02 | 2.57e+05 ± 1.59e+05 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 6.4 ± 4.3 | 15.6 ± 2.2 | 5.53e-02 ± 1.34e-02 |
| NN-tuned | test_bulk | 6642 | 9.48e-04 ± 8.00e-04 | 4.69e-02 ± 1.71e-02 | 1.05e-01 ± 9.27e-03 | 0.0 ± 0.0 | 8.2 ± 6.1 | 56.0 ± 19.0 | 90.3 ± 3.8 | 0.0 ± 0.0 | 8.15e-03 ± 2.01e-03 |
| NN-tuned | test_tail | 2608 | 1.17e-02 ± 8.30e-03 | 2.26e+02 ± 6.10e+01 | 9.95e+04 ± 1.89e+04 | 0.0 ± 0.0 | 0.0 ± 0.0 | 4.4 ± 3.5 | 48.1 ± 11.7 | 9.7 ± 0.7 | 1.83e-02 ± 5.81e-03 |
| RFF-ridge | test_bulk | 6642 | 3.15e-02 ± 3.74e-02 | 4.53e+00 ± 5.04e+00 | 6.31e+00 ± 5.59e+00 | 0.0 ± 0.0 | 1.3 ± 1.3 | 10.0 ± 7.6 | 40.1 ± 23.6 | 4.5 ± 5.2 | 3.35e-02 ± 2.39e-02 |
| RFF-ridge | test_tail | 2608 | 1.17e-01 ± 1.23e-01 | 4.71e+03 ± 7.15e+03 | 2.69e+06 ± 5.57e+06 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.8 ± 1.0 | 7.9 ± 7.2 | 19.5 ± 10.9 | 6.32e-02 ± 2.67e-02 |
| lookup-2D | test_bulk | 6642 | 2.44e-12 | 7.98e-09 | 3.89e-08 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 | 5.23e-07 |
| lookup-2D | test_tail | 2608 | 1.08e-09 | 3.09e-04 | 2.12e+02 | 93.8 | 98.5 | 99.2 | 99.5 | 0.5 | 6.50e-06 |
| lookup-nX | test_bulk | 6642 | 2.44e-12 | 7.98e-09 | 3.89e-08 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 | 5.23e-07 |
| lookup-nX | test_tail | 2608 | 1.08e-09 | 3.09e-04 | 2.12e+02 | 93.8 | 98.5 | 99.2 | 99.5 | 0.5 | 6.50e-06 |

### Reference points (the report's grid), and the supplementary score including the reference cliff_edge points

The `supp_ref_incl_cliff_*` rows are supplementary and were requested by the owner after the cliff_edge diagnosis
(STATUS 10a). The registered metric is the `ref_*` rows, which exclude those points. The reference flags 7 cliff_edge
points, but they are 5 distinct (n_X, L) pairs: two points of the main grid (`reference_lim.json`) are also
tail-grid points. All 5 lie in the tail region, so the bulk supplementary rows equal the `ref_bulk` rows by
construction.

| model | split | n | median | p99 | max | ≤1e-6 % | ≤1e-4 % | ≤1e-3 % | ≤1e-2 % | no key % | u-err median |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PQC-full | ref_bulk | 87 | 5.98e-03 ± 2.27e-03 | 5.67e-01 ± 5.28e-01 | 6.91e-01 ± 7.41e-01 | 0.0 ± 0.0 | 0.7 ± 0.6 | 15.4 ± 9.0 | 61.6 ± 9.7 | 0.5 ± 1.0 | 1.86e-02 ± 4.31e-03 |
| PQC-full | ref_tail | 175 | 4.86e-02 ± 1.77e-02 | 4.30e+02 ± 1.88e+02 | 5.85e+02 ± 2.80e+02 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.5 ± 0.6 | 15.2 ± 8.0 | 13.0 ± 3.7 | 3.94e-02 ± 1.03e-02 |
| PQC-full | supp_ref_incl_cliff_bulk | 87 | 5.98e-03 ± 2.27e-03 | 5.67e-01 ± 5.28e-01 | 6.91e-01 ± 7.41e-01 | 0.0 ± 0.0 | 0.7 ± 0.6 | 15.4 ± 9.0 | 61.6 ± 9.7 | 0.5 ± 1.0 | 1.86e-02 ± 4.31e-03 |
| PQC-full | supp_ref_incl_cliff_tail | 180 | 5.01e-02 ± 1.84e-02 | 7.97e+02 ± 3.83e+02 | 3.06e+03 ± 8.73e+02 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.4 ± 0.6 | 14.8 ± 7.8 | 15.4 ± 3.6 | 4.04e-02 ± 1.04e-02 |
| PQC-product | ref_bulk | 87 | 1.47e-02 ± 8.13e-03 | 1.07e+00 ± 8.67e-01 | 1.43e+00 ± 9.36e-01 | 0.0 ± 0.0 | 0.5 ± 1.0 | 4.8 ± 1.3 | 42.1 ± 12.0 | 2.5 ± 3.4 | 2.35e-02 ± 4.47e-03 |
| PQC-product | ref_tail | 175 | 5.53e-02 ± 1.84e-02 | 6.30e+02 ± 2.36e+02 | 8.47e+02 ± 4.70e+02 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 6.4 ± 4.2 | 14.1 ± 2.3 | 5.32e-02 ± 1.29e-02 |
| PQC-product | supp_ref_incl_cliff_bulk | 87 | 1.47e-02 ± 8.13e-03 | 1.07e+00 ± 8.67e-01 | 1.43e+00 ± 9.36e-01 | 0.0 ± 0.0 | 0.5 ± 1.0 | 4.8 ± 1.3 | 42.1 ± 12.0 | 2.5 ± 3.4 | 2.35e-02 ± 4.47e-03 |
| PQC-product | supp_ref_incl_cliff_tail | 180 | 5.65e-02 ± 1.89e-02 | 1.14e+03 ± 7.09e+02 | 2.77e+03 ± 1.47e+03 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 6.2 ± 4.1 | 16.4 ± 2.2 | 5.38e-02 ± 1.32e-02 |
| NN-tuned | ref_bulk | 87 | 8.88e-04 ± 8.33e-04 | 4.44e-02 ± 2.04e-02 | 5.25e-02 ± 2.96e-02 | 0.0 ± 0.0 | 9.2 ± 7.5 | 59.3 ± 20.8 | 87.4 ± 3.4 | 0.0 ± 0.0 | 7.54e-03 ± 2.31e-03 |
| NN-tuned | ref_tail | 175 | 1.22e-02 ± 7.54e-03 | 7.69e+01 ± 3.44e+01 | 1.29e+02 ± 5.63e+01 | 0.0 ± 0.0 | 0.0 ± 0.0 | 5.1 ± 5.6 | 47.2 ± 12.1 | 8.8 ± 1.3 | 1.72e-02 ± 5.62e-03 |
| NN-tuned | supp_ref_incl_cliff_bulk | 87 | 8.88e-04 ± 8.33e-04 | 4.44e-02 ± 2.04e-02 | 5.25e-02 ± 2.96e-02 | 0.0 ± 0.0 | 9.2 ± 7.5 | 59.3 ± 20.8 | 87.4 ± 3.4 | 0.0 ± 0.0 | 7.54e-03 ± 2.31e-03 |
| NN-tuned | supp_ref_incl_cliff_tail | 180 | 1.29e-02 ± 7.77e-03 | 1.75e+02 ± 5.22e+01 | 1.03e+03 ± 1.99e+02 | 0.0 ± 0.0 | 0.0 ± 0.0 | 5.0 ± 5.4 | 45.9 ± 11.7 | 11.3 ± 1.2 | 1.77e-02 ± 5.81e-03 |
| RFF-ridge | ref_bulk | 87 | 3.62e-02 ± 4.50e-02 | 4.43e+00 ± 5.20e+00 | 5.56e+00 ± 5.37e+00 | 0.0 ± 0.0 | 1.1 ± 1.6 | 8.7 ± 7.3 | 39.3 ± 27.6 | 6.7 ± 6.8 | 3.46e-02 ± 2.76e-02 |
| RFF-ridge | ref_tail | 175 | 1.31e-01 ± 1.58e-01 | 7.16e+02 ± 7.16e+02 | 2.67e+03 ± 3.16e+03 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.8 ± 1.2 | 9.0 ± 9.2 | 18.4 ± 13.3 | 7.02e-02 ± 4.41e-02 |
| RFF-ridge | supp_ref_incl_cliff_bulk | 87 | 3.62e-02 ± 4.50e-02 | 4.43e+00 ± 5.20e+00 | 5.56e+00 ± 5.37e+00 | 0.0 ± 0.0 | 1.1 ± 1.6 | 8.7 ± 7.3 | 39.3 ± 27.6 | 6.7 ± 6.8 | 3.46e-02 ± 2.76e-02 |
| RFF-ridge | supp_ref_incl_cliff_tail | 180 | 1.34e-01 ± 1.61e-01 | 4.33e+03 ± 7.24e+03 | 2.74e+04 ± 5.43e+04 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.8 ± 1.2 | 8.8 ± 8.9 | 20.7 ± 12.9 | 7.07e-02 ± 4.37e-02 |
| lookup-2D | ref_bulk | 87 | 1.23e-12 | 3.89e-09 | 5.87e-09 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 | 3.00e-07 |
| lookup-2D | ref_tail | 175 | 8.54e-10 | 7.66e-05 | 2.09e-04 | 95.4 | 99.4 | 100.0 | 100.0 | 0.0 | 5.67e-06 |
| lookup-2D | supp_ref_incl_cliff_bulk | 87 | 1.23e-12 | 3.89e-09 | 5.87e-09 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 | 3.00e-07 |
| lookup-2D | supp_ref_incl_cliff_tail | 180 | 1.05e-09 | 2.40e-04 | 8.08e-04 | 93.3 | 97.8 | 100.0 | 100.0 | 0.0 | 5.88e-06 |
| lookup-nX | ref_bulk | 87 | 1.23e-12 | 3.89e-09 | 5.87e-09 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 | 3.00e-07 |
| lookup-nX | ref_tail | 175 | 8.54e-10 | 7.66e-05 | 2.09e-04 | 95.4 | 99.4 | 100.0 | 100.0 | 0.0 | 5.67e-06 |
| lookup-nX | supp_ref_incl_cliff_bulk | 87 | 1.23e-12 | 3.89e-09 | 5.87e-09 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 | 3.00e-07 |
| lookup-nX | supp_ref_incl_cliff_tail | 180 | 1.05e-09 | 2.40e-04 | 8.08e-04 | 93.3 | 97.8 | 100.0 | 100.0 | 0.0 | 5.88e-06 |

### Held-out block sizes (interpolation in n_X)

| model | split | n | median | p99 | max | ≤1e-6 % | ≤1e-4 % | ≤1e-3 % | ≤1e-2 % | no key % | u-err median |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PQC-full | heldout_3e+06_bulk | 2623 | 4.36e-03 ± 1.14e-03 | 3.33e-02 ± 1.36e-02 | 3.46e-02 ± 1.60e-02 | 0.0 ± 0.0 | 1.1 ± 2.2 | 12.0 ± 6.9 | 75.5 ± 9.3 | 0.0 ± 0.0 | 1.97e-02 ± 3.56e-03 |
| PQC-full | heldout_5e+08_bulk | 2917 | 1.07e-02 ± 4.81e-03 | 5.67e-02 ± 2.91e-02 | 6.37e-02 ± 3.12e-02 | 0.0 ± 0.0 | 1.0 ± 2.0 | 12.2 ± 10.8 | 51.4 ± 19.3 | 0.0 ± 0.0 | 1.96e-02 ± 5.73e-03 |
| PQC-full | heldout_3e+06_tail | 480 | 4.24e-02 ± 2.21e-02 | 7.41e+02 ± 4.27e+02 | 2.40e+04 ± 1.39e+04 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 19.5 ± 11.5 | 12.2 ± 2.3 | 4.89e-02 ± 1.63e-02 |
| PQC-full | heldout_5e+08_tail | 480 | 5.16e-02 ± 1.68e-02 | 9.09e+02 ± 3.14e+02 | 8.66e+04 ± 2.99e+04 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.1 ± 0.2 | 17.0 ± 11.2 | 10.6 ± 1.2 | 5.63e-02 ± 1.19e-02 |
| PQC-product | heldout_3e+06_bulk | 2623 | 6.97e-02 ± 1.29e-01 | 6.62e-01 ± 1.29e+00 | 6.65e-01 ± 1.29e+00 | 0.0 ± 0.0 | 0.0 ± 0.0 | 4.5 ± 2.9 | 38.8 ± 23.2 | 6.5 ± 14.5 | 4.11e-02 ± 2.54e-02 |
| PQC-product | heldout_5e+08_bulk | 2917 | 2.17e-02 ± 6.35e-03 | 4.60e-01 ± 2.91e-01 | 5.65e-01 ± 5.15e-01 | 0.0 ± 0.0 | 0.0 ± 0.1 | 5.3 ± 2.1 | 34.0 ± 5.7 | 0.2 ± 0.4 | 2.62e-02 ± 2.47e-03 |
| PQC-product | heldout_3e+06_tail | 480 | 4.45e-02 ± 2.03e-02 | 1.10e+03 ± 5.09e+02 | 3.55e+04 ± 1.65e+04 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 9.1 ± 11.4 | 14.2 ± 2.6 | 5.72e-02 ± 1.34e-02 |
| PQC-product | heldout_5e+08_tail | 480 | 5.90e-02 ± 1.82e-02 | 1.33e+03 ± 5.77e+02 | 1.26e+05 ± 5.47e+04 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 9.8 ± 7.0 | 11.8 ± 1.8 | 7.07e-02 ± 1.81e-02 |
| NN-tuned | heldout_3e+06_bulk | 2623 | 1.06e-03 ± 9.14e-04 | 5.99e-03 ± 8.20e-03 | 6.21e-03 ± 8.09e-03 | 0.0 ± 0.0 | 3.0 ± 3.6 | 65.7 ± 31.6 | 96.2 ± 8.5 | 0.0 ± 0.0 | 1.13e-02 ± 2.07e-03 |
| NN-tuned | heldout_5e+08_bulk | 2917 | 1.37e-03 ± 1.86e-03 | 1.88e-02 ± 3.00e-02 | 1.90e-02 ± 3.00e-02 | 0.0 ± 0.0 | 8.2 ± 7.3 | 58.6 ± 33.3 | 90.3 ± 15.2 | 0.0 ± 0.0 | 7.46e-03 ± 2.35e-03 |
| NN-tuned | heldout_3e+06_tail | 480 | 5.52e-03 ± 2.67e-03 | 1.39e+02 ± 8.01e+01 | 4.33e+03 ± 2.59e+03 | 0.0 ± 0.0 | 0.0 ± 0.0 | 7.3 ± 6.6 | 67.0 ± 10.2 | 6.5 ± 1.3 | 1.40e-02 ± 3.98e-03 |
| NN-tuned | heldout_5e+08_tail | 480 | 9.92e-03 ± 9.86e-03 | 1.70e+02 ± 5.81e+01 | 1.59e+04 ± 5.60e+03 | 0.0 ± 0.0 | 0.0 ± 0.0 | 8.8 ± 11.0 | 54.0 ± 21.9 | 5.6 ± 0.7 | 2.01e-02 ± 1.04e-02 |
| RFF-ridge | heldout_3e+06_bulk | 2623 | 3.88e-02 ± 6.05e-02 | 4.60e-01 ± 7.65e-01 | 4.70e-01 ± 7.64e-01 | 0.0 ± 0.0 | 0.9 ± 1.4 | 17.3 ± 17.5 | 49.3 ± 27.1 | 2.1 ± 4.7 | 3.44e-02 ± 2.95e-02 |
| RFF-ridge | heldout_5e+08_bulk | 2917 | 5.52e-02 ± 6.21e-02 | 1.47e+00 ± 1.18e+00 | 1.49e+00 ± 1.18e+00 | 0.0 ± 0.0 | 0.9 ± 0.9 | 6.2 ± 4.3 | 26.1 ± 15.7 | 5.6 ± 7.1 | 4.70e-02 ± 3.52e-02 |
| RFF-ridge | heldout_3e+06_tail | 480 | 6.93e-02 ± 5.52e-02 | 1.94e+03 ± 1.69e+03 | 6.29e+04 ± 5.54e+04 | 0.0 ± 0.0 | 0.0 ± 0.0 | 2.8 ± 4.0 | 15.3 ± 14.8 | 16.1 ± 4.8 | 6.57e-02 ± 3.14e-02 |
| RFF-ridge | heldout_5e+08_tail | 480 | 5.55e-02 ± 2.18e-02 | 1.52e+03 ± 1.25e+03 | 1.44e+05 ± 1.18e+05 | 0.0 ± 0.0 | 0.3 ± 0.7 | 1.2 ± 2.6 | 9.4 ± 5.6 | 12.1 ± 3.4 | 7.10e-02 ± 1.75e-02 |
| lookup-2D | heldout_3e+06_bulk | 2623 | 2.03e-06 | 1.22e-05 | 1.25e-05 | 0.0 | 100.0 | 100.0 | 100.0 | 0.0 | 6.83e-04 |
| lookup-2D | heldout_5e+08_bulk | 2917 | 3.06e-05 | 5.81e-05 | 5.95e-05 | 0.0 | 100.0 | 100.0 | 100.0 | 0.0 | 2.27e-03 |
| lookup-2D | heldout_3e+06_tail | 480 | 4.43e-06 | 8.21e-04 | 4.72e+01 | 0.0 | 95.4 | 99.2 | 99.4 | 0.6 | 5.83e-04 |
| lookup-2D | heldout_5e+08_tail | 480 | 4.89e-05 | 3.97e-02 | 1.25e+02 | 0.0 | 78.5 | 93.1 | 97.9 | 0.6 | 1.77e-03 |

### Extrapolation (reported separately, never pooled)

| model | split | n | median | p99 | max | ≤1e-6 % | ≤1e-4 % | ≤1e-3 % | ≤1e-2 % | no key % | u-err median |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PQC-full | extrap_5e+03_bulk | 784 | 4.87e+00 ± 1.54e+00 | 1.49e+01 ± 5.56e+00 | 1.57e+01 ± 5.88e+00 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 100.0 ± 0.0 | 8.00e-02 ± 6.76e-03 |
| PQC-full | extrap_2e+09_bulk | 2948 | 5.70e-02 ± 3.56e-02 | 1.50e-01 ± 5.82e-02 | 1.69e-01 ± 5.40e-02 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 6.4 ± 7.9 | 0.0 ± 0.0 | 6.24e-02 ± 9.13e-03 |
| PQC-full | extrap_5e+03_tail | 432 | 2.93e+01 ± 7.93e+00 | 2.52e+04 ± 1.02e+04 | 5.56e+05 ± 2.28e+05 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 100.0 ± 0.0 | 1.30e-01 ± 1.36e-02 |
| PQC-full | extrap_2e+09_tail | 480 | 1.33e-01 ± 5.27e-02 | 6.36e+02 ± 1.19e+02 | 2.79e+04 ± 5.18e+03 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 10.8 ± 1.5 | 8.91e-02 ± 2.96e-02 |
| PQC-product | extrap_5e+03_bulk | 784 | 1.61e+00 ± 1.34e+00 | 4.34e+00 ± 1.19e+00 | 4.54e+00 ± 1.28e+00 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 60.5 ± 33.7 | 6.16e-02 ± 1.30e-02 |
| PQC-product | extrap_2e+09_bulk | 2948 | 6.12e-02 ± 4.99e-02 | 1.70e+00 ± 1.16e+00 | 1.72e+00 ± 1.17e+00 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.3 ± 0.5 | 15.7 ± 3.5 | 7.5 ± 7.2 | 4.40e-02 ± 1.08e-02 |
| PQC-product | extrap_5e+03_tail | 432 | 8.71e+00 ± 2.31e+00 | 1.33e+04 ± 6.81e+03 | 2.96e+05 ± 1.54e+05 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 100.0 ± 0.0 | 9.19e-02 ± 2.51e-02 |
| PQC-product | extrap_2e+09_tail | 480 | 7.97e-02 ± 3.95e-02 | 8.10e+02 ± 2.28e+02 | 3.54e+04 ± 9.91e+03 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.7 ± 0.9 | 10.8 ± 1.2 | 7.44e-02 ± 1.90e-02 |
| NN-tuned | extrap_5e+03_bulk | 784 | 7.11e-01 ± 6.83e-01 | 3.29e+00 ± 1.16e+00 | 3.58e+00 ± 1.30e+00 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 34.7 ± 24.6 | 6.23e-02 ± 1.04e-02 |
| NN-tuned | extrap_2e+09_bulk | 2948 | 2.52e-03 ± 3.01e-03 | 1.65e-02 ± 2.74e-02 | 1.67e-02 ± 2.73e-02 | 0.0 ± 0.0 | 0.0 ± 0.0 | 40.3 ± 36.5 | 91.3 ± 19.5 | 0.0 ± 0.0 | 1.86e-02 ± 7.61e-03 |
| NN-tuned | extrap_5e+03_tail | 432 | 1.32e+01 ± 6.82e+00 | 2.05e+04 ± 1.80e+04 | 4.54e+05 ± 4.03e+05 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 100.0 ± 0.0 | 1.21e-01 ± 2.82e-02 |
| NN-tuned | extrap_2e+09_tail | 480 | 1.41e-02 ± 1.20e-02 | 1.27e+02 ± 2.91e+01 | 5.52e+03 ± 1.33e+03 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 47.2 ± 12.6 | 5.5 ± 0.4 | 2.23e-02 ± 9.30e-03 |
| RFF-ridge | extrap_5e+03_bulk | 784 | 1.33e+01 ± 2.46e+01 | 3.54e+01 ± 4.27e+01 | 3.58e+01 ± 4.28e+01 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 73.7 ± 26.7 | 1.45e-01 ± 1.38e-01 |
| RFF-ridge | extrap_2e+09_bulk | 2948 | 2.80e-01 ± 2.81e-01 | 2.36e+00 ± 1.37e+00 | 2.40e+00 ± 1.40e+00 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.3 ± 0.6 | 3.4 ± 3.5 | 23.3 ± 18.4 | 1.48e-01 ± 1.29e-01 |
| RFF-ridge | extrap_5e+03_tail | 432 | 1.17e+02 ± 2.47e+02 | 2.44e+05 ± 5.29e+05 | 5.44e+06 ± 1.18e+07 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 100.0 ± 0.0 | 1.38e-01 ± 1.59e-01 |
| RFF-ridge | extrap_2e+09_tail | 480 | 4.46e-01 ± 3.22e-01 | 1.13e+03 ± 1.73e+03 | 4.78e+04 ± 7.28e+04 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 19.1 ± 15.8 | 2.15e-01 ± 1.10e-01 |
| lookup-2D | extrap_5e+03_bulk | 784 | 2.27e+00 | 6.39e+00 | 6.86e+00 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 7.51e-02 |
| lookup-2D | extrap_2e+09_bulk | 2948 | 8.68e-03 | 1.55e-02 | 1.61e-02 | 0.0 | 0.0 | 0.0 | 73.1 | 0.0 | 3.11e-02 |
| lookup-2D | extrap_5e+03_tail | 432 | 2.02e+01 | 1.94e+04 | 4.23e+05 | 0.0 | 0.0 | 0.0 | 0.0 | 100.0 | 1.43e-01 |
| lookup-2D | extrap_2e+09_tail | 480 | 2.66e-02 | 5.37e-01 | 8.61e+00 | 0.0 | 0.0 | 0.0 | 0.0 | 0.4 | 2.61e-02 |

### Crossover windows (±1 km around the 27 crossovers of the repaired labels; Amendments 2 and 4) and the drift window (listed)

| model | split | n | median | p99 | max | ≤1e-6 % | ≤1e-4 % | ≤1e-3 % | ≤1e-2 % | no key % | u-err median |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PQC-full | crossover_windows | 476 | 8.39e-03 ± 4.87e-03 | 1.08e-01 ± 3.14e-02 | 1.11e-01 ± 2.72e-02 | 0.0 ± 0.0 | 0.8 ± 1.2 | 13.2 ± 6.9 | 57.5 ± 9.0 | 0.0 ± 0.0 | 2.44e-02 ± 4.16e-03 |
| PQC-full | drift_window_listed | 2 | 2.68e-02 ± 2.52e-02 | 2.69e-02 ± 2.53e-02 | 2.69e-02 ± 2.53e-02 | 0.0 ± 0.0 | 0.0 ± 0.0 | 20.0 ± 44.7 | 40.0 ± 54.8 | 0.0 ± 0.0 | 2.93e-02 ± 1.78e-02 |
| PQC-product | crossover_windows | 476 | 5.29e-02 ± 6.23e-02 | 7.26e-01 ± 1.15e+00 | 7.56e-01 ± 1.20e+00 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.7 ± 0.4 | 24.3 ± 15.0 | 3.4 ± 7.5 | 4.00e-02 ± 1.20e-02 |
| PQC-product | drift_window_listed | 2 | 1.88e-02 ± 1.50e-02 | 1.89e-02 ± 1.50e-02 | 1.89e-02 ± 1.50e-02 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 40.0 ± 54.8 | 0.0 ± 0.0 | 2.76e-02 ± 1.19e-02 |
| NN-tuned | crossover_windows | 476 | 1.43e-03 ± 1.16e-03 | 7.84e-02 ± 1.02e-02 | 8.51e-02 ± 8.85e-03 | 0.0 ± 0.0 | 2.6 ± 2.8 | 47.3 ± 29.8 | 88.7 ± 1.1 | 0.0 ± 0.0 | 1.30e-02 ± 1.74e-03 |
| NN-tuned | drift_window_listed | 2 | 1.36e-03 ± 2.42e-03 | 1.36e-03 ± 2.42e-03 | 1.36e-03 ± 2.42e-03 | 0.0 ± 0.0 | 30.0 ± 44.7 | 80.0 ± 44.7 | 100.0 ± 0.0 | 0.0 ± 0.0 | 7.66e-03 ± 7.95e-03 |
| RFF-ridge | crossover_windows | 476 | 6.87e-02 ± 1.08e-01 | 9.96e-01 ± 1.08e+00 | 1.02e+00 ± 1.08e+00 | 0.0 ± 0.0 | 0.0 ± 0.1 | 14.2 ± 12.8 | 33.3 ± 21.8 | 4.5 ± 6.4 | 4.46e-02 ± 3.28e-02 |
| RFF-ridge | drift_window_listed | 2 | 2.08e-02 ± 2.73e-02 | 2.09e-02 ± 2.73e-02 | 2.09e-02 ± 2.73e-02 | 0.0 ± 0.0 | 0.0 ± 0.0 | 40.0 ± 54.8 | 60.0 ± 54.8 | 0.0 ± 0.0 | 3.63e-02 ± 3.41e-02 |
| lookup-2D | crossover_windows | 476 | 9.94e-06 | 8.46e-03 | 7.38e-02 | 29.8 | 77.5 | 81.9 | 99.4 | 0.0 | 1.13e-03 |
| lookup-2D | drift_window_listed | 2 | 8.70e-06 | 1.38e-05 | 1.39e-05 | 0.0 | 100.0 | 100.0 | 100.0 | 0.0 | 2.01e-03 |
| lookup-nX | crossover_windows | 188 | 1.46e-10 | 1.40e-02 | 7.38e-02 | 75.5 | 93.1 | 96.8 | 98.4 | 0.0 | 2.49e-06 |
| lookup-nX | drift_window_listed | 2 | 8.70e-06 | 1.38e-05 | 1.39e-05 | 0.0 | 100.0 | 100.0 | 100.0 | 0.0 | 2.01e-03 |

### Horizon error (km, predicted minus label; largest lattice L with R(p̂) > 0)

| model | 5000 | 1e+04 | 3.162e+04 | 1e+05 | 3.162e+05 | 1e+06 | 3e+06 | 3.162e+06 | 1e+07 | 3.162e+07 | 1e+08 | 3.162e+08 | 5e+08 | 1e+09 | 2e+09 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PQC-full | – | -4.6 ± 1.6 | -2.6 ± 0.4 | -2.2 ± 0.3 | -2.0 ± 0.4 | -1.9 ± 0.4 | -1.8 ± 0.3 | -1.8 ± 0.4 | -1.8 ± 0.3 | -1.8 ± 0.3 | -1.7 ± 0.3 | -1.6 ± 0.2 | -1.6 ± 0.2 | -1.6 ± 0.1 | -1.6 ± 0.2 |
| PQC-product | -56.1 ± 28.5 | -3.9 ± 0.6 | -3.0 ± 0.4 | -2.5 ± 0.3 | -2.2 ± 0.3 | -2.2 ± 0.5 | -2.2 ± 0.4 | -2.2 ± 0.4 | -2.0 ± 0.3 | -2.0 ± 0.3 | -2.0 ± 0.4 | -1.9 ± 0.3 | -1.8 ± 0.3 | -1.7 ± 0.2 | -1.6 ± 0.2 |
| NN-tuned | -43.5 ± 22.0 | -4.3 ± 1.0 | -2.1 ± 0.3 | -1.7 ± 0.2 | -1.4 ± 0.1 | -1.2 ± 0.1 | -1.0 ± 0.2 | -1.0 ± 0.2 | -0.9 ± 0.2 | -0.8 ± 0.1 | -0.8 ± 0.1 | -0.8 ± 0.1 | -0.8 ± 0.1 | -0.8 ± 0.1 | -0.9 ± 0.1 |
| RFF-ridge | -57.1 ± 21.9 | -3.0 ± 2.4 | -3.5 ± 1.6 | -3.1 ± 1.7 | -3.0 ± 1.9 | -2.8 ± 1.4 | -2.4 ± 0.7 | -2.4 ± 0.7 | -2.2 ± 0.5 | -2.3 ± 0.8 | -2.8 ± 1.7 | -2.3 ± 1.0 | -1.8 ± 0.5 | -1.9 ± 1.0 | -2.9 ± 2.4 |
| lookup-2D | – | -0.1 | -0.0 | -0.1 | -0.0 | -0.1 | -0.1 | -0.1 | -0.0 | -0.1 | -0.1 | -0.1 | -0.1 | -0.0 | -0.1 |
| lookup-nX | – | -0.1 | -0.0 | -0.1 | -0.0 | -0.1 | – | -0.1 | -0.0 | -0.1 | -0.1 | -0.1 | – | -0.0 | – |

### Secondary: the same test splits certified against the original labels

`results/qml_certify_final_a9ef3e3082_18624ccd048e/metrics.json`. The scored point sets differ: the repaired labels have 27 crossover windows
instead of 13, so more points fall inside a window and leave the test splits (dense PQC-full test_bulk n:
6642 repaired vs 6725 original).

| model | split | n | median | p99 | max | ≤1e-6 % | ≤1e-4 % | ≤1e-3 % | ≤1e-2 % | no key % | u-err median |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PQC-full | test_bulk | 6725 | 6.00e-03 ± 2.72e-03 | 5.48e-01 ± 4.87e-01 | 7.49e-01 ± 7.27e-01 | 0.0 ± 0.0 | 0.8 ± 0.9 | 14.9 ± 8.9 | 63.8 ± 9.7 | 0.2 ± 0.5 | 1.83e-02 ± 4.40e-03 |
| PQC-full | test_tail | 2608 | 4.67e-02 ± 1.22e-02 | 1.11e+03 ± 5.13e+02 | 3.11e+05 ± 9.07e+04 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.6 ± 0.5 | 15.1 ± 7.8 | 14.4 ± 2.6 | 4.16e-02 ± 9.89e-03 |
| PQC-product | test_bulk | 6725 | 1.44e-02 ± 8.06e-03 | 1.09e+00 ± 9.92e-01 | 1.62e+00 ± 1.04e+00 | 0.0 ± 0.0 | 0.1 ± 0.2 | 5.8 ± 2.3 | 43.6 ± 11.8 | 2.1 ± 3.5 | 2.40e-02 ± 5.35e-03 |
| PQC-product | test_tail | 2608 | 5.95e-02 ± 1.85e-02 | 1.54e+03 ± 8.04e+02 | 2.57e+05 ± 1.59e+05 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 6.4 ± 4.3 | 15.6 ± 2.2 | 5.53e-02 ± 1.34e-02 |
| NN-tuned | test_bulk | 6725 | 9.50e-04 ± 8.07e-04 | 4.78e-02 ± 1.68e-02 | 1.05e-01 ± 9.27e-03 | 0.1 ± 0.0 | 8.3 ± 6.2 | 55.9 ± 19.1 | 90.2 ± 3.8 | 0.0 ± 0.0 | 8.19e-03 ± 2.03e-03 |
| NN-tuned | test_tail | 2608 | 1.17e-02 ± 8.30e-03 | 2.26e+02 ± 6.10e+01 | 9.95e+04 ± 1.89e+04 | 0.0 ± 0.0 | 0.0 ± 0.0 | 4.4 ± 3.5 | 48.1 ± 11.7 | 9.7 ± 0.7 | 1.83e-02 ± 5.81e-03 |
| RFF-ridge | test_bulk | 6725 | 3.16e-02 ± 3.75e-02 | 4.50e+00 ± 5.04e+00 | 6.31e+00 ± 5.59e+00 | 0.0 ± 0.0 | 1.3 ± 1.3 | 10.0 ± 7.6 | 40.1 ± 23.5 | 4.5 ± 5.2 | 3.36e-02 ± 2.40e-02 |
| RFF-ridge | test_tail | 2608 | 1.17e-01 ± 1.23e-01 | 4.71e+03 ± 7.15e+03 | 2.69e+06 ± 5.57e+06 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.8 ± 1.0 | 7.9 ± 7.2 | 19.5 ± 10.9 | 6.32e-02 ± 2.67e-02 |
| lookup-2D | test_bulk | 6725 | 2.46e-12 | 8.87e-09 | 8.62e-05 | 99.9 | 100.0 | 100.0 | 100.0 | 0.0 | 5.29e-07 |
| lookup-2D | test_tail | 2608 | 1.08e-09 | 3.09e-04 | 2.12e+02 | 93.8 | 98.5 | 99.2 | 99.5 | 0.5 | 6.50e-06 |
| lookup-nX | test_bulk | 6725 | 2.46e-12 | 8.87e-09 | 8.62e-05 | 99.9 | 100.0 | 100.0 | 100.0 | 0.0 | 5.29e-07 |
| lookup-nX | test_tail | 2608 | 1.08e-09 | 3.09e-04 | 2.12e+02 | 93.8 | 98.5 | 99.2 | 99.5 | 0.5 | 6.50e-06 |

## Points where the cold solve is beaten (STATUS 12)

A cold solve (2^14 Sobol, top-16, seed 11) can miss the better of two optima next to a crossover. At these 21 points
a warm sweep label beats it (two-sided check at the 4746 gate points, plus two points checked
separately). The label is used as the reference there; it is the best known optimum, not a proven global one. A
point enters a reported number only through the groups listed; groups are recomputed for each label file.
Groups touched (repaired labels): crossover_windows (13), test_tail (1).

| n_X | L km | split | label source | label gain over cold | groups (repaired labels) | groups (original labels) | dense PQC-full regret, mean over seeds: repaired / original labels | cold-baseline regret (repaired labels) |
|---|---|---|---|---|---|---|---|---|
| 1e+04 | 117.8438 | test | backward | 6.20e-06 (separate check, STATUS 12) | test_tail | test_tail | 2.32e-01 / 2.32e-01 | 6.23e-06 |
| 3.16e+04 | 130.7500 | test | backward | 2.50e-05 (separate check, STATUS 12) | crossover_windows | test_bulk | 5.89e-02 / 5.89e-02 | 2.47e-05 |
| 1e+05 | 132.7500 | test | backward | 1.75e-05 | crossover_windows | test_bulk | 4.59e-02 / 4.59e-02 | -7.29e-15 |
| 1e+05 | 132.7812 | none | backward | 3.96e-05 | not scored / not evaluated | not scored / not evaluated | – / – | – |
| 1e+05 | 132.8125 | none | backward | 6.17e-05 | not scored / not evaluated | not scored / not evaluated | – / – | – |
| 3.16e+05 | 136.7500 | test | backward | 2.73e-05 | crossover_windows | test_bulk | 2.02e-02 / 2.02e-02 | 2.73e-05 |
| 3.16e+05 | 136.7812 | none | backward | 5.55e-05 | not scored / not evaluated | not scored / not evaluated | – / – | – |
| 3.16e+05 | 136.8125 | none | backward | 8.37e-05 | not scored / not evaluated | not scored / not evaluated | – / – | – |
| 3.16e+05 | 136.8438 | none | backward | 1.12e-04 | not scored / not evaluated | not scored / not evaluated | – / – | – |
| 3.16e+05 | 136.8750 | test | backward | 1.40e-04 | crossover_windows | test_bulk | 2.04e-02 / 2.03e-02 | 2.33e-15 |
| 3e+06 | 112.3438 | heldout | backfill | 5.76e-05 | crossover_windows | crossover_windows | 1.50e-03 / 1.50e-03 | 3.30e-15 |
| 3e+06 | 112.3750 | heldout | backfill | 1.21e-04 | crossover_windows | crossover_windows | 1.57e-03 / 1.57e-03 | 8.69e-16 |
| 3e+06 | 112.4062 | heldout | backfill | 1.84e-04 | crossover_windows | crossover_windows | 1.63e-03 / 1.63e-03 | 4.35e-15 |
| 3e+06 | 112.4375 | heldout | backfill | 2.47e-04 | crossover_windows | crossover_windows | 1.69e-03 / 1.69e-03 | -1.74e-16 |
| 3e+06 | 112.4688 | heldout | branch_switch | 3.10e-04 | crossover_windows | crossover_windows | 1.75e-03 / 1.75e-03 | 6.99e-16 |
| 3e+06 | 112.5000 | heldout | warm | 3.74e-04 | crossover_windows | crossover_windows | 1.82e-03 / 1.82e-03 | 1.23e-15 |
| 3e+06 | 139.1875 | heldout | backward | 2.83e-04 | crossover_windows | heldout_3e+06_bulk | 7.24e-03 / 6.96e-03 | -2.49e-15 |
| 3e+06 | 139.2188 | heldout | backward | 3.16e-04 | crossover_windows | heldout_3e+06_bulk | 7.24e-03 / 6.92e-03 | -4.99e-15 |
| 1e+08 | 91.8750 | test | backward | 1.67e-05 | crossover_windows | test_bulk | 2.71e-03 / 2.70e-03 | 1.67e-05 |
| 1e+08 | 92.0000 | train | backward | 8.07e-05 | not scored / not evaluated | not scored / not evaluated | – / – | – |
| 1e+08 | 92.1250 | val | backward | 1.45e-04 | not scored / not evaluated | not scored / not evaluated | – / – | – |

## Hybrid: surrogate + k Newton steps (dense; test splits)

| model (k=1) | split | n | median | p99 | max | ≤1e-6 % | ≤1e-4 % | ≤1e-3 % | ≤1e-2 % | no key % | u-err median |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PQC-full | test_bulk | 6642 | 8.41e-05 ± 8.38e-05 | 2.09e-01 ± 2.02e-01 | 4.98e-01 ± 3.88e-01 | 11.9 ± 6.0 | 56.1 ± 11.0 | 80.7 ± 5.9 | 91.7 ± 3.7 | 0.0 ± 0.0 | – |
| PQC-full | test_tail | 2608 | 7.00e-03 ± 5.76e-03 | 7.90e+01 ± 4.83e+01 | 2.60e+04 ± 6.51e+03 | 0.4 ± 0.3 | 13.8 ± 6.6 | 29.1 ± 10.2 | 58.0 ± 9.8 | 4.0 ± 0.8 | – |
| PQC-product | test_bulk | 6642 | 3.55e-04 ± 3.53e-04 | 9.56e-01 ± 1.46e+00 | 1.35e+00 ± 1.64e+00 | 5.5 ± 2.5 | 38.8 ± 11.0 | 67.0 ± 10.8 | 87.1 ± 6.5 | 1.3 ± 2.8 | – |
| PQC-product | test_tail | 2608 | 9.63e-03 ± 4.98e-03 | 1.16e+02 ± 6.37e+01 | 2.53e+04 ± 1.65e+04 | 0.0 ± 0.0 | 5.9 ± 3.2 | 20.9 ± 7.9 | 52.4 ± 11.4 | 4.6 ± 1.0 | – |
| NN-tuned | test_bulk | 6642 | 2.11e-06 ± 2.99e-06 | 2.46e-03 ± 1.25e-03 | 5.29e-02 ± 3.17e-02 | 48.0 ± 16.9 | 89.1 ± 5.3 | 96.9 ± 1.3 | 99.7 ± 0.3 | 0.0 ± 0.0 | – |
| NN-tuned | test_tail | 2608 | 2.60e-04 ± 4.46e-04 | 2.36e+01 ± 5.32e+00 | 1.02e+04 ± 1.79e+03 | 5.0 ± 2.8 | 48.0 ± 10.5 | 63.1 ± 7.9 | 82.2 ± 4.5 | 2.5 ± 0.2 | – |
| RFF-ridge | test_bulk | 6642 | 4.23e-03 ± 7.52e-03 | 7.27e+00 ± 1.00e+01 | 1.12e+01 ± 1.10e+01 | 8.5 ± 7.2 | 36.8 ± 23.0 | 57.7 ± 26.6 | 77.4 ± 22.3 | 3.3 ± 3.3 | – |
| RFF-ridge | test_tail | 2608 | 2.13e-02 ± 2.97e-02 | 5.09e+02 ± 8.10e+02 | 2.74e+05 ± 5.65e+05 | 0.7 ± 0.8 | 6.9 ± 5.2 | 20.2 ± 9.3 | 52.1 ± 16.8 | 6.6 ± 5.0 | – |
| lookup-2D | test_bulk | 6642 | 7.53e-16 | 6.58e-15 | 1.08e-14 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 | – |
| lookup-2D | test_tail | 2608 | 3.82e-15 | 4.61e-07 | 2.92e+01 | 99.1 | 99.4 | 99.5 | 99.5 | 0.3 | – |
| lookup-nX | test_bulk | 6642 | 7.74e-16 | 6.50e-15 | 1.21e-14 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 | – |
| lookup-nX | test_tail | 2608 | 3.90e-15 | 4.61e-07 | 2.92e+01 | 99.1 | 99.4 | 99.5 | 99.5 | 0.3 | – |
| model (k=2) | split | n | median | p99 | max | ≤1e-6 % | ≤1e-4 % | ≤1e-3 % | ≤1e-2 % | no key % | u-err median |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PQC-full | test_bulk | 6642 | 7.92e-08 ± 1.30e-07 | 1.04e-01 ± 1.72e-01 | 2.69e-01 ± 3.67e-01 | 72.6 ± 9.6 | 89.1 ± 3.5 | 93.9 ± 2.8 | 98.2 ± 1.4 | 0.0 ± 0.0 | – |
| PQC-full | test_tail | 2608 | 9.16e-05 ± 1.44e-04 | 8.53e+00 ± 4.00e+00 | 2.72e+03 ± 6.54e+02 | 26.5 ± 8.9 | 56.9 ± 10.5 | 74.1 ± 8.3 | 83.6 ± 3.0 | 1.5 ± 0.2 | – |
| PQC-product | test_bulk | 6642 | 8.26e-07 ± 1.22e-06 | 1.15e+00 ± 2.09e+00 | 1.60e+00 ± 2.43e+00 | 58.6 ± 11.1 | 83.0 ± 7.3 | 91.1 ± 6.3 | 96.4 ± 4.5 | 1.2 ± 2.6 | – |
| PQC-product | test_tail | 2608 | 1.11e-04 ± 9.24e-05 | 1.28e+01 ± 6.52e+00 | 2.65e+03 ± 1.63e+03 | 18.5 ± 7.5 | 51.4 ± 10.4 | 77.1 ± 6.9 | 82.8 ± 2.4 | 1.7 ± 0.4 | – |
| NN-tuned | test_bulk | 6642 | 7.49e-11 ± 1.50e-10 | 1.27e-05 ± 1.00e-05 | 3.61e-02 ± 3.27e-02 | 95.6 ± 2.2 | 99.6 ± 0.3 | 99.9 ± 0.2 | 100.0 ± 0.0 | 0.0 ± 0.0 | – |
| NN-tuned | test_tail | 2608 | 3.80e-07 ± 8.21e-07 | 9.19e-01 ± 1.01e-02 | 1.12e+03 ± 1.66e+02 | 61.5 ± 8.5 | 80.8 ± 6.0 | 86.9 ± 1.6 | 88.9 ± 1.0 | 0.9 ± 0.0 | – |
| RFF-ridge | test_bulk | 6642 | 1.50e-04 ± 3.15e-04 | 1.07e+01 ± 1.58e+01 | 1.69e+01 ± 1.72e+01 | 51.5 ± 27.5 | 73.4 ± 24.8 | 82.4 ± 19.6 | 91.5 ± 10.5 | 3.0 ± 2.9 | – |
| RFF-ridge | test_tail | 2608 | 1.37e-03 ± 2.75e-03 | 5.78e+01 ± 9.66e+01 | 3.19e+04 ± 6.67e+04 | 18.6 ± 8.7 | 51.2 ± 18.0 | 66.9 ± 20.2 | 76.6 ± 14.2 | 2.5 ± 2.1 | – |
| lookup-2D | test_bulk | 6642 | 6.50e-16 | 6.12e-15 | 1.03e-14 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 | – |
| lookup-2D | test_tail | 2608 | -1.57e-16 | 2.30e-12 | 4.72e+00 | 99.5 | 99.5 | 99.5 | 99.5 | 0.2 | – |
| lookup-nX | test_bulk | 6642 | 6.66e-16 | 5.88e-15 | 8.50e-15 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 | – |
| lookup-nX | test_tail | 2608 | -1.52e-16 | 2.22e-12 | 4.72e+00 | 99.5 | 99.5 | 99.5 | 99.5 | 0.2 | – |
| model (k=3) | split | n | median | p99 | max | ≤1e-6 % | ≤1e-4 % | ≤1e-3 % | ≤1e-2 % | no key % | u-err median |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PQC-full | test_bulk | 6642 | 2.42e-13 ± 5.19e-13 | 1.26e-01 ± 2.76e-01 | 2.19e-01 ± 4.25e-01 | 92.0 ± 3.0 | 97.0 ± 2.0 | 99.1 ± 0.7 | 99.5 ± 0.6 | 0.0 ± 0.0 | – |
| PQC-full | test_tail | 2608 | 5.11e-08 ± 1.10e-07 | 9.89e-01 ± 9.94e-04 | 3.27e+02 ± 7.55e+01 | 71.7 ± 9.6 | 83.4 ± 3.1 | 84.4 ± 2.8 | 85.0 ± 2.6 | 0.6 ± 0.1 | – |
| PQC-product | test_bulk | 6642 | 1.26e-11 ± 2.54e-11 | 1.31e+00 ± 2.43e+00 | 1.79e+00 ± 2.88e+00 | 87.7 ± 6.9 | 95.5 ± 4.9 | 96.8 ± 4.2 | 97.4 ± 3.8 | 1.1 ± 2.5 | – |
| PQC-product | test_tail | 2608 | 3.91e-08 ± 4.90e-08 | 9.87e-01 ± 1.65e-03 | 3.16e+02 ± 1.86e+02 | 73.8 ± 7.1 | 82.5 ± 2.8 | 83.3 ± 2.3 | 83.7 ± 2.3 | 0.7 ± 0.1 | – |
| NN-tuned | test_bulk | 6642 | 1.23e-15 ± 4.57e-16 | 5.89e-10 ± 6.73e-10 | 3.58e-02 ± 3.31e-02 | 99.9 ± 0.2 | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 0.0 ± 0.0 | – |
| NN-tuned | test_tail | 2608 | 1.69e-12 ± 3.74e-12 | 9.85e-01 ± 8.52e-04 | 1.40e+02 ± 2.00e+01 | 86.4 ± 2.0 | 88.8 ± 1.0 | 89.4 ± 0.8 | 89.8 ± 0.7 | 0.4 ± 0.0 | – |
| RFF-ridge | test_bulk | 6642 | 6.97e-07 ± 1.55e-06 | 1.37e+01 ± 2.06e+01 | 2.23e+01 ± 2.28e+01 | 78.8 ± 22.4 | 88.7 ± 14.0 | 93.4 ± 8.3 | 94.9 ± 6.1 | 3.0 ± 2.9 | – |
| RFF-ridge | test_tail | 2608 | 1.26e-05 ± 2.80e-05 | 6.61e+00 ± 1.02e+01 | 3.20e+03 ± 6.64e+03 | 65.4 ± 20.5 | 75.7 ± 15.0 | 78.2 ± 12.6 | 79.6 ± 11.3 | 1.0 ± 0.8 | – |
| lookup-2D | test_bulk | 6642 | 6.43e-16 | 6.11e-15 | 9.70e-15 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 | – |
| lookup-2D | test_tail | 2608 | -8.01e-16 | 5.29e-13 | 1.50e+00 | 99.5 | 99.5 | 99.5 | 99.5 | 0.1 | – |
| lookup-nX | test_bulk | 6642 | 6.56e-16 | 5.88e-15 | 8.50e-15 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 | – |
| lookup-nX | test_tail | 2608 | -8.75e-16 | 4.88e-13 | 1.50e+00 | 99.5 | 99.5 | 99.5 | 99.5 | 0.1 | – |

### Baselines for the two hybrid scenarios

| baseline | split | n | median | max | ≤1e-6 % | no key % | iterations (mean) |
|---|---|---|---|---|---|---|---|
| cold | test_bulk | 6642 | -1.35e-15 | 6.82e-15 | 100.0 | 0.0 | – |
| cold | test_tail | 2608 | -9.56e-15 | 6.23e-06 | 100.0 | 0.0 | – |
| cold | ref_bulk | 87 | 2.26e-15 | 1.19e-14 | 100.0 | 0.0 | – |
| cold | ref_tail | 175 | 9.86e-15 | 6.47e-13 | 100.0 | 0.0 | – |
| cold | heldout_3e+06_bulk | 2623 | -1.50e-15 | 9.83e-15 | 100.0 | 0.0 | – |
| cold | heldout_5e+08_bulk | 2917 | -1.22e-15 | 5.25e-15 | 100.0 | 0.0 | – |
| cold | heldout_3e+06_tail | 480 | -1.08e-14 | 1.32e-12 | 100.0 | 0.0 | – |
| cold | heldout_5e+08_tail | 480 | -7.18e-15 | 8.98e-13 | 100.0 | 0.0 | – |
| cold | extrap_5e+03_bulk | 784 | -2.49e-15 | 6.56e-15 | 100.0 | 0.0 | – |
| cold | extrap_2e+09_bulk | 2948 | -1.15e-15 | 3.28e-15 | 100.0 | 0.0 | – |
| cold | extrap_5e+03_tail | 432 | -9.05e-15 | 3.48e-12 | 100.0 | 0.0 | – |
| cold | extrap_2e+09_tail | 480 | -6.03e-15 | 4.78e-13 | 100.0 | 0.0 | – |
| warm | test_bulk | 6642 | 3.19e-16 | 1.26e-14 | 100.0 | 0.0 | 2.4 |
| warm | test_tail | 2608 | -1.32e-15 | 6.23e-06 | 100.0 | 0.0 | 4.7 |
| warm | ref_bulk | 87 | 4.00e-15 | 8.35e-15 | 100.0 | 0.0 | 2.2 |
| warm | ref_tail | 175 | 1.53e-14 | 3.73e-13 | 100.0 | 0.0 | 4.1 |
| warm | heldout_3e+06_bulk | 2622 | 0.00e+00 | 9.99e-15 | 100.0 | 0.0 | 2.6 |
| warm | heldout_5e+08_bulk | 2916 | 0.00e+00 | 8.77e-15 | 100.0 | 0.0 | 2.1 |
| warm | heldout_3e+06_tail | 480 | 4.50e-16 | 8.11e-13 | 100.0 | 0.0 | 3.4 |
| warm | heldout_5e+08_tail | 480 | 5.58e-16 | 8.27e-13 | 100.0 | 0.0 | 3.0 |
| warm | extrap_5e+03_bulk | 783 | 0.00e+00 | 1.06e-14 | 100.0 | 0.0 | 2.3 |
| warm | extrap_2e+09_bulk | 2947 | 0.00e+00 | 6.24e-15 | 100.0 | 0.0 | 1.9 |
| warm | extrap_5e+03_tail | 432 | -1.74e-15 | 3.36e-12 | 100.0 | 0.0 | 5.5 |
| warm | extrap_2e+09_tail | 480 | -4.63e-16 | 4.37e-13 | 100.0 | 0.0 | 2.9 |

**Scenario (ii) neighbour distance (decision (a) at STOP 2).**
- The preregistration's text says the neighbour is "0.5 km away in the bulk and 0.125 km in the tail".
- The registered rule is the nearest lower training node. For test points that node is 0.25 or 0.375 km away in the
  bulk, and 0.0625 or 0.094 km in the tail. The reason is the warm-sweep node spacing: training nodes every 0.5 km
  in the bulk and every 0.125 km in the tail, with test points offset between them.
- Held-out and extrapolated block sizes have no training nodes. There the neighbour is the previous lattice label
  (0.125 km below 100 km, 1/32 km above).
- Measured on the test splits (repaired-label groups): test_bulk: median 0.25 km; most common 0.25 km (50 %), 0.375 km (50 %); test_tail: median 0.0938 km; most common 0.0625 km (50 %), 0.0938 km (50 %), 0.4688 km (0 %), 0.2188 km (0 %).
- Over all evaluation points pooled by region (test, ref, held-out and extrapolated together, so this mixes the
  rules): median bulk 0.125 km, tail 0.0625 km.

## Latency and cost (latency: dense seed-0 weights; training time: all seeds, per hardware)

| model | params | train s, CPU (seed 0, Apple M2 Pro) | train s, GPU (A100; mean ± sd, seeds 1-4) | batch-1 device µs | batch-1 end-to-end µs | batch-1024 throughput /s | backend |
|---|---|---|---|---|---|---|---|
| PQC-full | 240 | 1196.0 | 162.5 ± 3.4 | 278.9 | 298.1 | 2.13e+04 | default.qubit (analytic), jitted |
| PQC-product | 180 | 224.0 | 48.8 ± 2.1 | 104.9 | 119.8 | 1.48e+05 | default.qubit (analytic), jitted |
| NN-tuned | 257 | 2.0 | 2.9 ± 1.0 | 6.1 | 18.5 | 9.35e+06 | JAX MLP, jitted |
| RFF-ridge | 145 | not recorded | 36.3 ± 1.3 | 6.8 | 19.9 | 5.12e+06 | JAX, jitted |
| lookup-2D | 4727 | – | – | 24.2 | 38.2 | 5.48e+05 | JAX, jitted (padded rows) |

- Batch-1 Newton steps from a surrogate prediction, end-to-end median: k=1 844 µs, k=2 1307 µs, k=3 1765 µs.
- Batch-1 cold solve (optimize, 2^12 Sobol, top-8): 28736 µs median (p90 31560 µs).
- Batch-1 warm solve from the nearest lower training node: 1499 µs median, 2 iterations.
- PQC on hardware: shots per prediction for a standard error ε in every ⟨Z_i⟩ (median over the query set): eps_0.01 → 9.71e+03, eps_0.001 → 9.71e+05, eps_0.0001 → 9.71e+07. One circuit per prediction.

The batch-1 timings above are from the dedicated quiet re-timing, which **supersedes** the batch-1 timings recorded
inside the two certification runs. Those were single passes taken during a heavy job and differ by up to about 2×; they
are kept below as a documented caveat, not as results. Quiet re-timing `results/qml_retime_35aba07eec_431340851504/retime.json`: 5 full passes; AC power; 1-min load average before 1.54 (limit 2.0), after each pass 3.23, 3.23, 3.23, 3.23, 3.23 (1-min averages that include the timing process itself; pass 1 also compiles); no other qml or benchmark job running.

| batch-1 end-to-end median, µs | **quiet re-timing** (median over passes; min–max) | repaired-label certification run | original-label certification run |
|---|---|---|---|
| k = 1 Newton steps | **844 (827–854)** | 1165 | 2522 |
| k = 2 Newton steps | **1307 (1304–1314)** | 1357 | 3802 |
| k = 3 Newton steps | **1765 (1758–1768)** | 6350 | 5057 |
| cold solve | **28736 (28556–29075)** | 34903 | 72524 |
| warm solve | **1499 (1492–1545)** | 1897 | 5111 |

The usefulness verdicts below use the quiet re-timing. Recomputed with either certification run's timings,
no (i)/(ii) verdict changes.
- Batched baselines over all 21137 evaluation points (wall clock, this laptop): cold
  841 s and warm 97 s in this certification;
  1344 s and 142 s in the original-label certification.
  The cold solve is the same computation in both, so wall-clock timings of this size vary by tens of percent between runs.
- Training time: CPU (seed 0, the stopped run on the Mac) and GPU (seeds 1-4, Leonardo A100) are separate columns and are
  never pooled.

## Usefulness under Section 7 (dense; test splits)

The definitions are **post hoc**: the preregistration does not define "regret class" or "comparable latency".
- **(b)** The model has a lower median regret than lookup-2D, a higher % within 1e-6, and a batch-1 end-to-end
  latency of at most 10× lookup-2D's (38 µs).
- **(i) and (ii)** For some k ≤ 3, the hybrid's % within 1e-6 is at least the cold solve's (i) or the warm solve's
  (ii). Its latency (surrogate batch-1 end-to-end + k-step batch-1 end-to-end) is also below the cold solve's
  (28736 µs) or the warm solve's (1499 µs) batch-1 end-to-end.

The two latencies are added across environments. Transfer between `.venv-qml` and `.venv` is not included.

| model | split | median regret | ≤1e-6 % | batch-1 e2e µs | (b) beats lookup-2D at ≤10× its latency | (i) hybrid reaches cold-solve class faster | (ii) hybrid reaches warm-solve class faster |
|---|---|---|---|---|---|---|---|
| PQC-full | test_bulk | 6.00e-03 | 0.0 | 298.1 | no | no | no |
| PQC-full | test_tail | 4.67e-02 | 0.0 | 298.1 | no | no | no |
| PQC-product | test_bulk | 1.43e-02 | 0.0 | 119.8 | no | no | no |
| PQC-product | test_tail | 5.95e-02 | 0.0 | 119.8 | no | no | no |
| NN-tuned | test_bulk | 9.48e-04 | 0.0 | 18.5 | no | no | no |
| NN-tuned | test_tail | 1.17e-02 | 0.0 | 18.5 | no | no | no |
| RFF-ridge | test_bulk | 3.15e-02 | 0.0 | 19.9 | no | no | no |
| RFF-ridge | test_tail | 1.17e-01 | 0.0 | 19.9 | no | no | no |
| lookup-2D | test_bulk | 2.44e-12 | 100.0 | 38.2 | n/a (is the floor) | yes: k1 (882 µs, 100.0 %), k2 (1345 µs, 100.0 %), k3 (1803 µs, 100.0 %) | yes: k1 (882 µs, 100.0 %), k2 (1345 µs, 100.0 %) |
| lookup-2D | test_tail | 1.08e-09 | 93.8 | 38.2 | n/a (is the floor) | no | no |

## Sparse training (Amendment 3; 2 km labels; its own tables, never pooled)

| model | split | n | median | p99 | max | ≤1e-6 % | ≤1e-4 % | ≤1e-3 % | ≤1e-2 % | no key % | u-err median |
|---|---|---|---|---|---|---|---|---|---|---|---|
| PQC-full | test_bulk | 6642 | 3.44e-03 ± 1.25e-03 | 1.46e-01 ± 8.65e-02 | 5.23e-01 ± 8.30e-01 | 0.0 ± 0.0 | 2.1 ± 1.9 | 25.9 ± 9.1 | 77.1 ± 6.5 | 0.0 ± 0.1 | 1.32e-02 ± 2.47e-03 |
| PQC-full | test_tail | 2608 | 7.72e-02 ± 6.83e-03 | 2.52e+03 ± 1.17e+03 | 1.08e+06 ± 8.92e+05 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 2.4 ± 1.7 | 19.6 ± 3.3 | 6.36e-02 ± 4.95e-03 |
| PQC-full | ref_bulk | 87 | 3.67e-03 ± 1.49e-03 | 1.30e-01 ± 8.48e-02 | 1.48e-01 ± 1.03e-01 | 0.0 ± 0.0 | 1.8 ± 1.7 | 25.5 ± 10.4 | 74.5 ± 6.6 | 0.0 ± 0.0 | 1.33e-02 ± 2.51e-03 |
| PQC-full | ref_tail | 175 | 7.14e-02 ± 7.18e-03 | 7.83e+02 ± 2.34e+02 | 1.26e+03 ± 5.89e+02 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 3.0 ± 2.1 | 18.6 ± 4.1 | 6.29e-02 ± 5.33e-03 |
| PQC-full | heldout_3e+06_bulk | 2623 | 4.22e-03 ± 1.38e-03 | 3.49e-02 ± 1.35e-02 | 3.78e-02 ± 1.63e-02 | 0.0 ± 0.0 | 2.3 ± 4.0 | 22.0 ± 12.1 | 76.5 ± 8.9 | 0.0 ± 0.0 | 1.66e-02 ± 3.07e-03 |
| PQC-full | heldout_5e+08_bulk | 2917 | 4.05e-03 ± 1.33e-03 | 5.87e-02 ± 2.67e-02 | 6.42e-02 ± 2.53e-02 | 0.0 ± 0.0 | 1.8 ± 2.4 | 21.7 ± 8.4 | 77.3 ± 12.1 | 0.0 ± 0.0 | 1.40e-02 ± 2.45e-03 |
| PQC-full | heldout_3e+06_tail | 480 | 5.14e-02 ± 2.67e-02 | 1.58e+03 ± 9.36e+02 | 5.12e+04 ± 3.01e+04 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 6.9 ± 8.3 | 16.3 ± 3.6 | 5.89e-02 ± 1.64e-02 |
| PQC-full | heldout_5e+08_tail | 480 | 8.71e-02 ± 1.71e-02 | 1.58e+03 ± 4.19e+02 | 1.50e+05 ± 3.95e+04 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 13.1 ± 1.4 | 8.85e-02 ± 1.02e-02 |
| PQC-product | test_bulk | 6642 | 1.79e-02 ± 2.35e-02 | 1.14e+00 ± 1.34e+00 | 2.21e+00 ± 2.55e+00 | 0.0 ± 0.0 | 0.4 ± 0.3 | 9.8 ± 6.2 | 50.1 ± 20.7 | 2.5 ± 3.7 | 2.38e-02 ± 1.18e-02 |
| PQC-product | test_tail | 2608 | 9.10e-02 ± 2.86e-02 | 2.11e+03 ± 7.57e+02 | 6.10e+05 ± 6.00e+05 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 1.4 ± 0.8 | 17.8 ± 1.7 | 6.63e-02 ± 6.78e-03 |
| PQC-product | ref_bulk | 87 | 1.68e-02 ± 2.05e-02 | 1.28e+00 ± 1.58e+00 | 1.69e+00 ± 2.04e+00 | 0.0 ± 0.0 | 0.2 ± 0.5 | 9.4 ± 5.4 | 49.0 ± 19.9 | 2.8 ± 3.9 | 2.41e-02 ± 1.23e-02 |
| PQC-product | ref_tail | 175 | 8.38e-02 ± 2.35e-02 | 7.42e+02 ± 2.37e+02 | 1.04e+03 ± 2.60e+02 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 1.8 ± 1.2 | 16.6 ± 1.7 | 6.61e-02 ± 6.87e-03 |
| PQC-product | heldout_3e+06_bulk | 2623 | 4.60e-02 ± 8.62e-02 | 5.73e-01 ± 1.18e+00 | 5.79e-01 ± 1.18e+00 | 0.0 ± 0.0 | 0.3 ± 0.4 | 7.3 ± 6.8 | 49.6 ± 23.8 | 3.0 ± 6.8 | 3.94e-02 ± 3.27e-02 |
| PQC-product | heldout_5e+08_bulk | 2917 | 3.46e-02 ± 4.25e-02 | 9.51e-01 ± 1.22e+00 | 9.59e-01 ± 1.23e+00 | 0.0 ± 0.0 | 0.6 ± 0.6 | 7.0 ± 4.2 | 37.6 ± 17.0 | 8.4 ± 12.5 | 2.65e-02 ± 1.08e-02 |
| PQC-product | heldout_3e+06_tail | 480 | 8.86e-02 ± 5.62e-02 | 1.36e+03 ± 3.50e+02 | 4.42e+04 ± 1.12e+04 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 15.5 ± 1.6 | 7.72e-02 ± 2.45e-02 |
| PQC-product | heldout_5e+08_tail | 480 | 9.19e-02 ± 3.59e-02 | 1.65e+03 ± 5.89e+02 | 1.57e+05 ± 5.52e+04 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 2.2 ± 3.1 | 13.1 ± 1.6 | 9.02e-02 ± 1.42e-02 |
| NN-tuned | test_bulk | 6642 | 5.42e-04 ± 1.99e-04 | 2.77e-02 ± 1.33e-02 | 8.57e-02 ± 4.63e-02 | 0.0 ± 0.0 | 12.7 ± 6.8 | 64.9 ± 7.2 | 95.7 ± 1.3 | 0.0 ± 0.0 | 6.03e-03 ± 8.72e-04 |
| NN-tuned | test_tail | 2608 | 2.10e-02 ± 6.30e-03 | 3.01e+02 ± 4.28e+01 | 1.19e+05 ± 4.06e+04 | 0.0 ± 0.0 | 0.0 ± 0.0 | 3.6 ± 3.4 | 39.6 ± 4.8 | 10.6 ± 0.7 | 2.29e-02 ± 2.42e-03 |
| NN-tuned | ref_bulk | 87 | 5.06e-04 ± 1.62e-04 | 2.41e-02 ± 1.47e-02 | 4.26e-02 ± 3.44e-02 | 0.0 ± 0.0 | 14.0 ± 7.6 | 66.7 ± 7.8 | 94.7 ± 2.1 | 0.0 ± 0.0 | 5.88e-03 ± 1.05e-03 |
| NN-tuned | ref_tail | 175 | 1.83e-02 ± 6.38e-03 | 1.02e+02 ± 2.39e+01 | 1.68e+02 ± 3.88e+01 | 0.0 ± 0.0 | 0.0 ± 0.0 | 2.6 ± 1.9 | 40.7 ± 5.4 | 9.4 ± 1.0 | 2.13e-02 ± 2.78e-03 |
| NN-tuned | heldout_3e+06_bulk | 2623 | 5.86e-04 ± 1.29e-04 | 3.59e-03 ± 2.09e-03 | 3.82e-03 ± 2.39e-03 | 0.0 ± 0.0 | 5.2 ± 4.1 | 74.4 ± 8.7 | 100.0 ± 0.0 | 0.0 ± 0.0 | 9.18e-03 ± 1.82e-03 |
| NN-tuned | heldout_5e+08_bulk | 2917 | 4.43e-04 ± 3.06e-04 | 5.95e-03 ± 2.73e-03 | 6.02e-03 ± 2.80e-03 | 0.0 ± 0.0 | 16.8 ± 18.1 | 63.8 ± 8.5 | 100.0 ± 0.0 | 0.0 ± 0.0 | 5.76e-03 ± 9.87e-04 |
| NN-tuned | heldout_3e+06_tail | 480 | 1.05e-02 ± 3.89e-03 | 1.78e+02 ± 5.45e+01 | 5.64e+03 ± 1.76e+03 | 0.0 ± 0.0 | 0.0 ± 0.0 | 6.5 ± 9.9 | 49.9 ± 6.1 | 7.1 ± 0.9 | 2.06e-02 ± 3.51e-03 |
| NN-tuned | heldout_5e+08_tail | 480 | 1.52e-02 ± 5.07e-03 | 2.29e+02 ± 3.06e+01 | 2.18e+04 ± 2.93e+03 | 0.0 ± 0.0 | 0.0 ± 0.0 | 8.3 ± 9.1 | 46.4 ± 3.9 | 6.1 ± 0.3 | 2.80e-02 ± 5.01e-03 |
| RFF-ridge | test_bulk | 6642 | 2.41e-02 ± 2.56e-02 | 2.95e+00 ± 3.58e+00 | 5.46e+00 ± 3.92e+00 | 0.0 ± 0.0 | 1.9 ± 2.3 | 11.9 ± 11.0 | 46.4 ± 25.3 | 4.2 ± 5.4 | 3.09e-02 ± 1.97e-02 |
| RFF-ridge | test_tail | 2608 | 2.53e-01 ± 2.46e-01 | 1.38e+04 ± 1.26e+04 | 1.68e+07 ± 2.51e+07 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 3.8 ± 3.1 | 29.6 ± 15.3 | 8.62e-02 ± 4.26e-02 |
| RFF-ridge | ref_bulk | 87 | 2.69e-02 ± 2.60e-02 | 2.91e+00 ± 3.59e+00 | 3.17e+00 ± 3.81e+00 | 0.0 ± 0.0 | 1.6 ± 1.9 | 12.2 ± 9.8 | 46.2 ± 25.9 | 5.7 ± 6.5 | 3.23e-02 ± 2.11e-02 |
| RFF-ridge | ref_tail | 175 | 2.29e-01 ± 2.14e-01 | 2.17e+03 ± 1.71e+03 | 6.66e+03 ± 1.01e+04 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 5.6 ± 4.4 | 28.1 ± 15.8 | 8.43e-02 ± 4.42e-02 |
| RFF-ridge | heldout_3e+06_bulk | 2623 | 1.45e-02 ± 1.07e-02 | 7.93e-02 ± 4.13e-02 | 8.47e-02 ± 4.56e-02 | 0.0 ± 0.0 | 1.5 ± 1.8 | 10.0 ± 11.8 | 44.6 ± 29.8 | 0.0 ± 0.0 | 3.05e-02 ± 1.44e-02 |
| RFF-ridge | heldout_5e+08_bulk | 2917 | 3.28e-02 ± 3.08e-02 | 4.43e-01 ± 7.84e-01 | 4.44e-01 ± 7.84e-01 | 0.0 ± 0.0 | 1.0 ± 0.9 | 9.6 ± 10.7 | 31.1 ± 18.4 | 0.8 ± 1.9 | 3.17e-02 ± 1.51e-02 |
| RFF-ridge | heldout_3e+06_tail | 480 | 2.18e-01 ± 2.98e-01 | 5.01e+03 ± 7.73e+03 | 1.61e+05 ± 2.47e+05 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 2.5 ± 3.7 | 22.6 ± 15.0 | 8.33e-02 ± 2.56e-02 |
| RFF-ridge | heldout_5e+08_tail | 480 | 1.40e-01 ± 1.07e-01 | 4.08e+03 ± 4.85e+03 | 3.80e+05 ± 4.47e+05 | 0.0 ± 0.0 | 0.0 ± 0.0 | 0.0 ± 0.0 | 4.5 ± 8.1 | 18.5 ± 9.4 | 9.03e-02 ± 2.63e-02 |
| lookup-2D | test_bulk | 6642 | 3.45e-10 | 2.12e-06 | 3.89e-04 | 97.8 | 100.0 | 100.0 | 100.0 | 0.0 | 6.13e-06 |
| lookup-2D | test_tail | 2608 | 3.46e-05 | 5.69e+01 | 9.27e+03 | 8.4 | 61.1 | 79.0 | 89.2 | 3.5 | 1.18e-03 |
| lookup-2D | ref_bulk | 87 | 5.20e-11 | 5.19e-07 | 7.47e-07 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 | 1.84e-06 |
| lookup-2D | ref_tail | 175 | 2.71e-05 | 8.17e+00 | 3.53e+01 | 10.9 | 65.1 | 81.7 | 91.4 | 1.1 | 9.89e-04 |
| lookup-2D | heldout_3e+06_bulk | 2623 | 2.18e-06 | 1.23e-05 | 2.98e-04 | 0.0 | 99.6 | 100.0 | 100.0 | 0.0 | 6.88e-04 |
| lookup-2D | heldout_5e+08_bulk | 2917 | 3.08e-05 | 5.70e-05 | 5.84e-05 | 0.0 | 100.0 | 100.0 | 100.0 | 0.0 | 2.27e-03 |
| lookup-2D | heldout_3e+06_tail | 480 | 5.48e-05 | 7.89e-02 | 6.25e+01 | 0.0 | 57.3 | 78.1 | 89.4 | 0.6 | 1.71e-03 |
| lookup-2D | heldout_5e+08_tail | 480 | 1.18e-04 | 8.72e+01 | 9.10e+03 | 0.0 | 41.7 | 76.5 | 90.2 | 4.2 | 1.91e-03 |
| lookup-nX | test_bulk | 6642 | 3.45e-10 | 2.12e-06 | 3.89e-04 | 97.8 | 100.0 | 100.0 | 100.0 | 0.0 | 6.13e-06 |
| lookup-nX | test_tail | 2608 | 3.46e-05 | 5.69e+01 | 9.27e+03 | 8.4 | 61.1 | 79.0 | 89.2 | 3.5 | 1.18e-03 |
| lookup-nX | ref_bulk | 87 | 5.20e-11 | 5.19e-07 | 7.47e-07 | 100.0 | 100.0 | 100.0 | 100.0 | 0.0 | 1.84e-06 |
| lookup-nX | ref_tail | 175 | 2.71e-05 | 8.17e+00 | 3.53e+01 | 10.9 | 65.1 | 81.7 | 91.4 | 1.1 | 9.89e-04 |

## RFF-ridge: seed-to-seed spread

The dense RFF-ridge median test_bulk regret varies by more than 10× between seeds. The per-seed diagnostics
(`results/qml_rffdiag_a08b8d5e0b_eba620de9c21/rff_seeds.json`, recomputed from the saved frequencies and
coefficients) point to the construction, not to a pipeline failure:

| variant | seed | hardware | λ | median abs frequency | train MSE (z) | val MSE (z) | val MSE over λ grid (min–max) | test_bulk median regret |
|---|---|---|---|---|---|---|---|---|
| dense | 0 | cpu | not saved | 1.32 | 3.75e-03 | 3.80e-03 | 3.80e-03–3.80e-03 | 8.74e-03 |
| dense | 1 | gpu | 1e-10 | 2.74 | 4.34e-02 | 4.33e-02 | 4.33e-02–4.33e-02 | 9.42e-02 |
| dense | 2 | gpu | 1e-06 | 1.16 | 3.41e-03 | 3.43e-03 | 3.43e-03–3.95e-03 | 5.70e-03 |
| dense | 3 | gpu | 1e-10 | 1.08 | 5.03e-03 | 5.06e-03 | 5.06e-03–5.12e-03 | 1.09e-02 |
| dense | 4 | gpu | 1e-10 | 1.88 | 1.17e-02 | 1.16e-02 | 1.16e-02–1.16e-02 | 3.79e-02 |
| sparse | 0 | gpu | 1e-10 | 1.04 | 2.01e-03 | 3.81e-03 | 3.81e-03–4.74e-03 | 2.65e-03 |
| sparse | 1 | gpu | 1e-10 | 2.53 | 2.69e-02 | 3.68e-02 | 3.68e-02–3.68e-02 | 6.44e-02 |
| sparse | 2 | gpu | 1e-10 | 1.51 | 3.03e-03 | 5.76e-03 | 5.76e-03–6.37e-03 | 6.14e-03 |
| sparse | 3 | gpu | 1e-10 | 2.68 | 1.78e-02 | 2.93e-02 | 2.93e-02–2.93e-02 | 3.39e-02 |
| sparse | 4 | gpu | 1e-10 | 1.42 | 5.89e-03 | 1.11e-02 | 1.11e-02–1.11e-02 | 1.35e-02 |

- Each seed draws only 14 random frequencies (random {-1, 0, 1} combinations of its own trained PQC input scalings).
  Dense seed 1 drew mostly high frequencies (median |frequency| 2.74, against
  1.08–1.88
  for the other dense seeds), and underfits.
- The underfit is visible at training time: its training MSE equals its validation MSE (no overfitting) and both are
  3.7–12.7× the other dense seeds'. Validation MSE varies by at most 0.00025 % across the
  λ grid for seed 1, so λ selection is not the cause (seed 1 chose λ = 1e-10; seeds with
  the same λ: s3, s4), and F^T F is well conditioned (condition number 3.9e+02).
- Across the dense seeds, the ranking by training MSE equals the ranking by test regret:
  True.
- So the spread is a property of the RFF-ridge as preregistered (14 random features per seed), not a fault of one
  run. The mean ± sd over seeds in the tables includes seed 1 unchanged.

## Fourier spectrum of the trained PQC-full (selected configuration, seed 0, from tuning)

- Encoding gates per input: 30.
- Largest reachable frequency: u_L 12.35, v 11.48
  (sum of |trained input scalings|).
- The RFF ridge samples this spectrum with D/2 = 14 frequencies. Its result is in the tables as
  `RFF-ridge`.

## Figures for the paper

These figures were built earlier for slides; the slide decks are no longer produced (owner decision, 2026-11-09), and
the figures are used as paper figures instead. Each one is listed with what it was drawn from.

- `results/qml_certify_final_b983055317_6518a9e98390/regret_cdf_dense.png`: Regret CDF of the surrogate predictions (no Newton steps), test bulk and tail, dense training, all seeds. Source: `results/qml_certify_final_b983055317_6518a9e98390/metrics.json` (repaired labels).
- `results/qml_certify_final_b983055317_6518a9e98390/regret_cdf_sparse.png`: The same for sparse training (Amendment 3). Source: `results/qml_certify_final_b983055317_6518a9e98390/metrics.json`.
- `qml/slides/assets/showcase_predictions.png`: Predicted vs optimal settings across distance, and the % of the optimal key rate each prediction delivers; n_X = 1e6 test points and the held-out n_X = 5e8; PQC-full, NN-tuned, lookup-2D, seed 0. Source: `qml/slides/make_showcase.py`, repaired labels and the primary certification.
- `qml/slides/assets/params_across_distance.png`: Optimal settings across distance for n_X = 1e4, 1e6, 1e9 (labels only). Source: `qml/slides/make_showcase.py`, repaired labels.
- `qml/slides/assets/params_by_blocksize.png`: All five optimal settings across distance, one panel per block size, with every isolated jump marked. Source: `qml/slides/make_params_grid.py --labels <repaired>`.
- `qml/slides/assets/pqc_circuit_8q_labelled.png`: One layer of the selected PQC-full circuit (8 qubits; repeated 6 times; 5 answer qubits read out). Source: `qml/slides/draw_circuit.py` (.venv-qml, PennyLane).

## Limitations

- Circuits are simulated analytically (exact expectation values). Hardware shot noise and gate errors are not
  included; see the shot estimate above and Phase 5a.
- One block-size range [1e4, 1e9] and two inputs. Extended inputs are Phase 5b.
- The latency comparison crosses two environments and one CPU machine (`.venv-qml` jax 0.10.2; `.venv` jax 0.9.0).
- **cliff_edge.** For label points the registered rule flags 0 points. The reference's 7 flags (5 distinct
  points) are converged optima (STATUS 10a); the supplementary rows show the effect of including them.
- **Selection.** One validation split and one seed per configuration; 2000 steps for every family. Other budgets may
  rank the families differently.

## Sources

- selection `results/qml_select_7218993cb5_feab976f711d/selection.json`
- tuning `results/qml_tune_7218993cb5_32f0e26b35fd/tune.json`
- final `results/qml_final_07c50abb3c_6d14876dd876/final.json`
- latency `results/qml_latency_35aba07eec_23413b61099e/latency.json`
- metrics, primary (repaired labels) `results/qml_certify_final_b983055317_6518a9e98390/metrics.json`
- metrics, secondary (original labels) `results/qml_certify_final_a9ef3e3082_18624ccd048e/metrics.json`
- labels `qml/data/repaired/labels_repaired_b983055317_9dd20e4718fe.npz`; repair `results/qml_repair_b983055317_9dd20e4718fe`
- batch-1 timing (headline) `results/qml_retime_35aba07eec_431340851504/retime.json`
- RFF-ridge seed diagnostics `results/qml_rffdiag_a08b8d5e0b_eba620de9c21/rff_seeds.json`
- STATUS 12 checks `results/qml_coldcheck_e007a41265_cd1f8920532d/two_sided.json`, `results/qml_suboptcheck_b983055317_f5d7b9715fea/check.json`
- figures: `results/qml_certify_final_b983055317_6518a9e98390/regret_cdf_dense.png`, `results/qml_certify_final_b983055317_6518a9e98390/regret_cdf_sparse.png`

No quantum advantage is claimed.
