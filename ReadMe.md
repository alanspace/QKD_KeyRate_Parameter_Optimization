# Optimal Parameters for Finite-Key Decoy-State BB84 (Lim et al. 2014)

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![Framework: JAX](https://img.shields.io/badge/Framework-JAX-blueviolet.svg)
![Build Status](https://github.com/alanspace/QKD_KeyRate_Parameter_Optimization/actions/workflows/ci.yml/badge.svg)

This repository computes, verifies and benchmarks the parameters that maximise the secret key rate of finite-key
decoy-state BB84 under the security bounds of Lim, Curty, Walenta, Xu and Zbinden,
[PRA 89, 022307 (2014)](https://arxiv.org/abs/1311.7129).

- **Parameters:** signal and decoy intensities μ1, μ2 (μ3 = 2·10⁻⁴ fixed), their probabilities, and the basis
  probability q_x.
- **Range:** fiber length L ∈ [0, 200] km and block size n_X ∈ [10⁴, 10⁹].

It began as an MSc project on neural-network (NN) surrogates for this optimisation. It now provides a verified
optimizer and reference: the yardstick for the next step, a benchmark of classical-NN and quantum-machine-learning
surrogates.

**Report:** [`result_report/final_report.tex`](result_report/final_report.tex) (November 2026; the PDF builds
locally, see [`result_report/README.md`](result_report/README.md)). It supersedes every earlier report, slide deck
and README version.

## Results (preset `lim`, measured; see the report for definitions and sources)

**Physics model.** An audit against the paper found a defect in the earlier key-rate code: it used the X-basis
detection count where Eq. (4) needs the Z-basis error count m_Z. It also found four further departures from the
paper (e_mis, afterpulsing p_ap, ε_sec = κℓ, and the clamping of n⁻/m⁻).

The corrected model:
- agrees with an independent float64/Decimal implementation to ~10⁻¹³;
- reproduces the paper's three stated numbers to within 8 %:

| | this work | Lim et al. 2014 |
|---|---|---|
| R(L=0, n_X=10⁹) [bits/pulse] | 9.20·10⁻³ (−8 %) | ~10⁻² |
| R(100 km, 10⁹) / R(100 km, 10⁷) | 1.79 (+2 %) | ~1.75 |
| maximum distance at n_X = 10⁴ (ℓ > 0) | 130.4 km (−3 %) | ~135 km |

Maximum distance (largest L with key length ℓ > 0), by block size:

| n_X | 10⁴ | 10⁵ | 10⁶ | 10⁷ | 10⁸ | 10⁹ |
|---|---|---|---|---|---|---|
| km | 130.44 | 160.61 | 172.62 | 178.67 | 182.09 | 184.16 |

**Optimizer accuracy.** Regret is relative to an independently cross-checked reference; points where the key rate
is discontinuous (`cliff_edge`) are excluded.

| method | within 10⁻⁶ (bulk / near-horizon tail) | max regret (bulk / tail) |
|---|---|---|
| new optimizer, warm sweep | 100 % / 100 % | 1.0·10⁻¹³ / 9.5·10⁻¹³ |
| new optimizer, cold start | 100 % / 100 % | 9.1·10⁻¹⁴ / 6.5·10⁻¹³ |
| lookup table (2 km nodes; simplest surrogate) | 81.6 % / 24.6 % | 2.2·10⁻² / 23 (no key) |
| earlier dual-annealing run ¹ | 3.7 % / 0 % | 1.26 / 0.89 |
| earlier Hessian_FoR run (former NN labels) ¹ | 81.7 % / 0 % | 96 / 36 |

¹ Scored on the earlier physics they were optimised for (preset `legacy`).

**Parameter structure.** The optimal parameters vary smoothly with L, apart from two switches between competing
local optima while the key is still positive: n_X = 10⁵ near 130.5 km and n_X = 10⁶ near 118.5 km. The optimizer
detects, logs and backfills both. The earlier runs jumped repeatedly at 148–157 km.

**Cost** (Apple M2 Pro, CPU, float64; compile time excluded):

| method | time per point |
|---|---|
| earlier dual-annealing settings, cold, one point | 5.8–17.6 s |
| new optimizer, cold, vectorised batch of 126 points | 40 ms |
| new optimizer, warm start from a solution 0.5 km away, one point | 1.2 ms (device) / 1.5 ms (end-to-end) |
| lookup table | 8 µs / 10 µs |

**Withdrawn.** All numbers in earlier versions of this README and in the earlier reports and slides are withdrawn.
This includes ">6,000× speed-up", "47–60× higher key rate", "+5 km", "129× stability", "5.9 M×" and all NN accuracy
figures. [`CLAIMS.md`](CLAIMS.md) gives the status of each claim, and Appendix A of the report covers the 2025
thesis.

## Repository layout

| path | contents |
|---|---|
| `src/qkd_opt/` | **Current code.** Physics (`physics_lim.py`), presets (`presets.py`), feasible reparameterisation (`reparam.py`), optimizer (`optimize.py`), provenance stamps |
| `src/qkd/` | Earlier physics (preset `legacy`); kept unchanged for comparison |
| `Optimization/reference/` | Trusted reference builder, independent SciPy cross-check, reference JSONs |
| `Optimization/benchmark/` | Accuracy benchmark, timing benchmark, earlier-run loader |
| `Optimization/analysis/` | Physics ablation, cliff test, dense scan, γ report, local-optimality check, comparisons |
| `Optimization/report/` | `make_numbers.py` and `make_figures.py`, which generate every number and figure of the report |
| `Optimization/run_all.py` | Runs the whole pipeline and records each output in `results/MANIFEST.json` |
| `results/` | Generated outputs, each stamped with git commit, config hash, physics hash, hardware and power state |
| `result_report/` | Current report (`final_report.tex`); earlier documents under `superseded/` |
| `tests/` | 34 unit tests, including an independent re-implementation of the bounds |
| `Optimization/run_scripts/`, `Optimization/archive/`, `Optimization/notebooks/`, `NeuralNetwork/`, `Analysis/`, `Project_Report/` | Earlier code, notebooks, NN and the 2025 thesis; kept as historical record (see `REVIEW.md`) |

Further documentation:
- [`PHYSICS_DIFF.md`](PHYSICS_DIFF.md): presets, unprinted model choices, and differences from the paper.
- [`REVIEW.md`](REVIEW.md): review of the earlier optimizers.
- [`STATUS.md`](STATUS.md): run log with every measured value and the open issues.
- [`NN_RETRAIN_PLAN.md`](NN_RETRAIN_PLAN.md): plan for retraining the NN.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install "jax[cpu]"          # Linux + NVIDIA GPU: pip install "jax[cuda12]"
```

Use float64 throughout. On Apple silicon set `JAX_PLATFORMS=cpu`, because Metal has no float64. The NN code runs
in a separate environment (`requirements-nn.txt`, e.g. `.venv-nn`); the two environments exchange data only
through files.

## Usage

```python
from src.qkd_opt import get_preset
from src.qkd_opt.optimize import optimize, optimize_batch, optimize_sweep

res = optimize(100.0, 1e9, get_preset("lim"))   # one point (L = 100 km, n_X = 1e9), cold start
```

| step | command | output |
|---|---|---|
| everything | `python Optimization/run_all.py` (`--only STEP`, `--parallel N`) | `results/MANIFEST.json`, `results/NUMBERS.md`, `result_report/generated_numbers.tex` |
| trusted reference | `python Optimization/reference/build_reference.py --preset lim [--tail]` | `Optimization/reference/reference_<preset>[_tail].json` |
| independent cross-check | `python Optimization/reference/crosscheck_scipy.py --preset lim [--tail]` | `results/crosscheck_<preset>…/CROSSCHECK.md` |
| benchmark | `python Optimization/benchmark/run_benchmark.py --preset lim` | `results/benchmark_<preset>_…/RESULTS.md` |
| timing | `python Optimization/benchmark/timing_benchmark.py --preset lim` | `results/timing_<preset>_…/TIMING.md` |
| physics ablation and paper targets | `python Optimization/analysis/physics_diff.py` | `results/physics_diff_…/PHYSICS_DIFF_TABLES.md` |
| report numbers and figures | `python Optimization/report/make_numbers.py`; `python Optimization/report/make_figures.py` | `result_report/generated_numbers.tex`; `results/figures/…` |
| tests | `python -m unittest discover tests` | |

The full reference rebuild takes about 1 h on a laptop. Run long jobs on AC power, or on a cloud machine.

## Next steps

1. **Surrogate benchmark (classical NN vs quantum ML).** This will be a separate project. Section 7 of the report
   fixes the protocol in advance:
   - labels from the new optimizer;
   - test points held out from the training grid;
   - regret of the re-evaluated key rate, scored separately in bulk and tail;
   - the lookup table and the warm-started solver as baselines;
   - latency including shots and circuit evaluations for QML;
   - at least 5 seeds per model.
2. **Cloud compute.** Label generation, any dual-annealing dataset regeneration, and NN training and inference
   (about 20 min) are planned on cloud CPU/GPU instances, so that long runs do not stall a laptop.
3. **Open issues** are listed in [`STATUS.md`](STATUS.md) and Section 6.3 of the report.

## Citation

```bibtex
@mastersthesis{leung2025mlqkd,
  author     = {Leung, Shek Lun},
  title      = {Machine Learning for Quantum Key Distribution Network Optimization},
  school     = {KTH Royal Institute of Technology},
  year       = {2025},
  supervisor = {Svanberg, Erik and Foletto, Giulio and Adya, Vaishali},
  examiner   = {Gallo, Katia}
}
```

Physics: C. C. W. Lim, M. Curty, N. Walenta, F. Xu, H. Zbinden, *Concise security bounds for practical decoy-state
quantum key distribution*, Phys. Rev. A 89, 022307 (2014).
