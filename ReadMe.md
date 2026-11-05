# Machine Learning for Quantum Key Distribution Network Optimization

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)
![Framework: JAX](https://img.shields.io/badge/Framework-JAX-blueviolet.svg)
![Build Status](https://github.com/alanspace/QKD_KeyRate_Parameter_Optimization/actions/workflows/ci.yml/badge.svg)

This repository contains the optimization pipeline for the BB84 QKD protocol, investigating the use of neural networks (NNs) to accelerate parameter optimization. It supports both a "Legacy" NumPy-based baseline (Tier 1), a modern JAX-accelerated suite (Tier 3), and a Neural Network surrogate model.

## ⚡️ Interactive Live Dashboard

Experience the power of real-time AI optimization through our interactive dashboard.

[**Launch Live App on Streamlit Cloud**](https://appapppy-jcrvkdozjbapjtbgujxutu.streamlit.app/)

## Abstract

Optimizing parameters is crucial for maximizing the performance of Quantum Key Distribution (QKD) systems. This study investigates the efficacy of neural networks (NNs) as a high-speed alternative to Dual Annealing (DA) for determining optimal operational parameters (signal/decoy intensities `μk`, probabilities `Pμk`, basis choice `Px`) for the finite-key decoy-state BB84 protocol. We demonstrate that a trained NN can predict near-optimal parameters with high accuracy, achieving a **>6,000-fold practical speedup** over traditional legacy solvers.

## Key Results

- **📉 Rate Improvement:** **~47x - 60x higher key rate** at long distances (180 km+).
- **📏 Range Extension:** **+5.0 km** extended secure communication distance.
- **⚡ Implementation Speed:** **>6,000-fold practical speedup** (End-to-End Latency).
- **🎯 High Accuracy:** The NN predictions closely match the numerically optimized ground truth.

## Directory Structure

*   **`Optimization/`**: Main project folder for numerical optimization.
    *   **`run_scripts/`**: Executable scripts for different optimization tiers.
        *   `tier1_legacy_baseline.py`: **Baseline (Tier 1)**. Pure NumPy/SciPy `dual_annealing`. Slow but accurate.
        *   `tier3_global_jax.py`: **Sequential Adaptive (Tier 3)**. JAX-accelerated. The "Gold Standard" for smooth parameters.
        *   `tier3_jax_segmented.py`: **Segmented Adaptive**. Faster parallel execution with L-splitting.
        *   `tier2_local.py`: **Fast Local Search**. For quick verification.
        *   `plot_results.py`: Tool to generate plots from JSON results.
    *   **`outputs/`**: Centralized storage for all run results.
        *   `tier1_legacy_baseline/`: Logs and JSONs from Tier 1 runs.
        *   `global_jax/`: Logs and JSONs from Tier 3 runs.
    *   **`notebooks/`**: Jupyter notebooks (Archived/Research).
    *   **`tools/`**: Helper utilities (Timing reports, etc.).
*   **`src/`**: Shared physics library (`qkd.model`, `qkd.model_numpy`).
*   **`NeuralNetwork/`**: Neural network architecture, training notebooks, and pre-trained models.
*   **`Analysis/`**: Validation notebooks for the analytical model.

## Installation

1.  **Create Environment**:
    ```bash
    conda create -n qkd-opt python=3.10
    conda activate qkd-opt
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install JAX (OS-Specific)**:
    *   **Mac (Apple Silicon) / Linux / Windows (CPU)**: 
        ```bash
        pip install jax[cpu]
        ```
    *   **Linux (NVIDIA GPU)**:
        ```bash
        pip install jax[cuda12]
        ```

## Optimizer rework: `src/qkd_opt` (JAX-native, branch `optimizer-jax-rework`)

> **Audit notice.** The numeric claims in the Abstract, Key Results and Benchmarks sections of this
> README, and in `result_report/` and `assets/Optimization_Speedup_Math.*`, are audited in
> [`CLAIMS.md`](CLAIMS.md). Use the status shown there. The only numbers to rely on are those produced
> by the scripts below, whose outputs are in `results/` and carry the git commit and config hash.

* **Physics presets** (`src/qkd_opt/presets.py`). Every result is labelled with one:
  * `lim` (default): Lim et al. 2014 finite-key decoy-state bounds with η_sys in the misalignment term. It
    reproduces the paper within ~10 %, not exactly.
  * `lim_literal`: the printed η_ch. Diagnostic only.
  * `legacy`: the physics of `src/qkd`, evaluated by `physics_lim` with legacy switches.
  * Differences and unprinted model choices are in [`PHYSICS_DIFF.md`](PHYSICS_DIFF.md).
* **Optimizer** (`src/qkd_opt/optimize.py`):
  * Constraints hold by construction through an unconstrained reparameterisation.
  * A Sobol global stage (jit + vmap) is followed by a batched exact-Hessian Newton solver in float64.
  * Continuation along L uses periodic global branch checks, logs every branch switch, and backfills after a switch.
  * The horizon is defined as l > 0 and found by bisection.
  * Every point carries a status flag and a certificate: gradient norm, Hessian eigenvalues, active bounds, branch id.
  * It runs on CPU or GPU with no code change. On Apple silicon use `JAX_PLATFORMS=cpu`, because Metal has no float64.

```python
from src.qkd_opt import get_preset
from src.qkd_opt.optimize import optimize, optimize_batch, optimize_sweep
res = optimize(100.0, 1e9, get_preset("lim"))            # one point, cold start
```

| step | command | output |
|---|---|---|
| **everything** (references, tail references, cross-checks, physics diff, benchmarks, analyses, timing, tests, report numbers) | `python Optimization/run_all.py` | `results/MANIFEST.json`, `results/NUMBERS.md`, `result_report/generated_numbers.tex` |
| trusted reference | `python Optimization/reference/build_reference.py --preset lim [--tail]` | `Optimization/reference/reference_<preset>[_tail].json` |
| independent cross-check | `python Optimization/reference/crosscheck_scipy.py --preset lim [--tail]` | `results/crosscheck_<preset>…/CROSSCHECK.md` |
| benchmark | `python Optimization/benchmark/run_benchmark.py --preset lim` | `results/benchmark_<preset>_<commit>_<cfg>/RESULTS.md` + figures |
| timing | `python Optimization/benchmark/timing_benchmark.py --preset lim` | `results/timing_<preset>_<commit>_<cfg>/TIMING.md` |
| analyses | `python Optimization/analysis/analyses.py --preset lim` | `results/analyses_<preset>_<commit>_<cfg>/ANALYSES.md` |
| physics diff and paper targets | `python Optimization/analysis/physics_diff.py` | `results/physics_diff_<commit>_<cfg>/PHYSICS_DIFF_TABLES.md` |
| tests | `python -m unittest discover tests` | |

The report (`result_report/results_report.tex`) imports `generated_numbers.tex`, so its numbers are never
typed by hand. The NN surrogate runs in a separate environment (`.venv-nn`, `requirements-nn.txt`);
see [`NN_RETRAIN_PLAN.md`](NN_RETRAIN_PLAN.md).

The previous optimizer variants are archived, unmodified, in `Optimization/archive/jax_variants_2026-02/`.
See `REVIEW.md` for why.

## Usage: Optimization Pipeline (New Scripts)

For production-grade data generation and benchmarking, use the Python scripts in `Optimization/run_scripts/`.

### 1. Running the Baseline (Tier 1)
Pure NumPy implementation using `dual_annealing`. Useful for verifying results without JAX dependencies.
```bash
python Optimization/run_scripts/tier1_legacy_baseline.py
```
*   **Output**: `Optimization/outputs/tier1_legacy_baseline/`

### 2. Running the JAX Optimizer (Tier 3)
The state-of-the-art Adaptive JAX optimizer. Fastest and most robust.
```bash
python Optimization/run_scripts/tier3_global_jax.py
```
*   **Output**: `Optimization/outputs/global_jax/`

## Usage: Research Workflow (Notebooks)

Follow these steps to reproduce the full research study, including Model Verification and Neural Network Training.

### 1. Verification of the Analytical Model
**Notebook:** `Analysis/BB84_Parameters_2014_Analysis_Jax.ipynb`
*   Verifies the core QKD simulation.
*   Calculates Secret Key Rate (SKR) using *fixed* parameters to ensure correct exponential decay.

### 2. Neural Network Training and Evaluation
**Notebook:** `NeuralNetwork/neural_network_updated.ipynb`
*   Trains the PyTorch model using the dataset generated by the Optimization steps.
*   Evaluates accuracy and generates relative error plots.
*   Saves the trained model to `NeuralNetwork/models/`.

## Benchmarks: Speedup Evolution

We benchmarked the Neural Network against traditional solvers.

| Solver Type | Time per Point | Speedup Factor |
| :--- | :--- | :--- |
| **Legacy (Dual Annealing)** | $> 10.0$ s | 1x (Baseline) |
| **Modern (JAX/SciPy)** | $0.01 - 0.05$ s | $1,000\times$ |
| **Neural Network** | $0.000002$ s | **$>6,000\times$ - $140,000\times$** |

### Data Generation Speedup
Comparison of the background training data generation process:

| Feature | Old Method (Baseline) | New Method (JAX Adaptive) | Improvement |
| :--- | :--- | :--- | :--- |
| **Runtime (6000 pts)** | ~1 hr 30 min | ~2 min | **~45x Faster** ⚡ |
| **Algorithm** | `scipy.dual_annealing` | `jax.grad` + `Adaptive L-BFGS-B` | **Gradient-based** |
| **Quality** | Noisy (Staircase artifacts) | Smooth (Physically realistic) | **High Stability** |

## Limitations and Future Work

### Current Limitations
1.  **Prediction Accuracy Near Physical Limits**: <5% relative error in useful ranges, but higher near zero-key rate (negligible impact).
2.  **Static Channel Assumption**: Trained for static fiber conditions.

### Future Research Directions
1.  **Physics-Informed Neural Networks (PINNs)**: Incorporate rate equations into the loss function.
2.  **Real-Time Adaptation**: Deploy on FPGA/Edge devices for dynamic scenarios.
3.  **Transfer Learning**: Extend to MDI-QKD or Twin-Field QKD.

## Citation

If you use this work in your research, please cite the original project:

```bibtex
@mastersthesis{leung2024mlqkd,
  author       = {Leung, Shek Lun},
  title        = {Machine Learning for Quantum Key Distribution Network Optimization},
  school       = {KTH Royal Institute of Technology},
  year         = {2024},
  supervisor   = {Svanberg, Erik and Foletto, Giulio and Adya, Vaishali},
  examiner     = {Gallo, Katia}
}
```

This work is based on the analytical model presented in:
Lim, C. C. W., et al. (2014). "Concise security bounds for practical decoy-state quantum key distribution". Physical Review A, 89(2), 022307.
