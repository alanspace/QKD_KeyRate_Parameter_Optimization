# Implementation Plan: Skeptic-Proof Upgrades

We will implement the final 3 technical proofs to address global optimality, feature importance, and reproducibility.

## 1. Feature Importance Audit
- **Script**: `verify_report_metrics.py`
- **Logic**: Calculate sensitivity of NN output w.r.t inputs: $S_i = |\partial y / \partial x_i|$.
- **Expectation**: Sensitivity to Fiber Length ($x_0$) and Misalignment ($x_2$) should be dominant.

## 2. Error Distribution Visual
- **Script**: `generate_error_histogram.py`
- **Logic**: Load test set -> Predict -> Calc Relative Error -> Plot Histogram (Bell Curve).
- **Output**: `assets/error_histogram.png`.

## 3. Report Update
- Add "Global Optimality" section (Multi-start L-BFGS-B).
- Add "Feature Sensitivity" section.
- Add "Reproducibility" statement.
- Include `error_histogram.png`.
