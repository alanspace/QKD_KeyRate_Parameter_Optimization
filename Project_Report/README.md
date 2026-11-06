# MSc thesis material (May 2025): superseded

This folder holds the MSc thesis report and defence slides, *Machine Learning for Quantum Key Distribution Network
Optimization*. They are kept as the historical record.

**Their numerical results are superseded by `../result_report/final_report.pdf` (November 2026). Do not cite
them.** Appendix A of that report gives the details. In brief:

- **Physics.** The thesis model omits afterpulsing (p_ap = 0) and fixes ε_sec at 1e-10. The paper uses
  p_ap = 0.04 and ε_sec = κℓ. These two differences alone shorten the n_X = 1e4 horizon by about 22 km.
- **Horizons.** The thesis horizons (~145–180 km) use an R ≈ 1e-8 threshold, not ℓ = 0.
- **Labels.** The dual-annealing labels were never checked for optimality.
- **Claims.** "±6 % SKR error" and "270× speed-up" are not reproduced.
- **Still valid.** The background chapters (QKD, optimisation, neural networks) remain usable as background.
