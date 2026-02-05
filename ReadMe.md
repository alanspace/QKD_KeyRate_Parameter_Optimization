# QKD Key Rate Parameter Optimization

## Overview
This repository contains the optimization pipeline for the BB84 QKD protocol. It is structured to support both a "Legacy" NumPy-based baseline (Tier 1) and a modern JAX-accelerated suite (Tier 3).

## Directory Structure

*   **`Optimization/`**: Main project folder.
    *   **`run_scripts/`**: Executable scripts for different optimization tiers.
        *   `tier1_legacy_baseline.py`: **Baseline (Tier 1)**. Pure NumPy/SciPy `dual_annealing`. Slow but accurate.
        *   `tier3_global_jax.py`: **Sequential Adaptive (Tier 3)**. JAX-accelerated. The "Gold Standard" for smooth parameters.
        *   `tier3_jax_segmented.py`: **Segmented Adaptive**. Faster parallel execution with L-splitting.
        *   `tier2_local.py`: **Fast Local Search**. For quick verification.
        *   `plot_results.py`: Tool to generate plots from JSON results.
    *   **`outputs/`**: Centralized storage for all run results.
        *   `tier1_legacy_baseline/`: Logs and JSONs from Tier 1 runs.
        *   `global_jax/`: Logs and JSONs from Tier 3 runs.
        *   `notebook_comparison/`: Comparison plots against fixed parameters.
    *   **`notebooks/`**: Jupyter notebooks (Archived).
    *   **`tools/`**: Helper utilities (Timing reports, etc.).
*   **`src/`**: Shared physics library (`qkd.model`, `qkd.model_numpy`).

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
    *   **Mac/Linux/Windows (CPU)**: 
        ```bash
        pip install jax[cpu]
        ```
    *   **Linux (NVIDIA GPU)**:
        ```bash
        pip install jax[cuda12]
        ```

## Usage

### Running the Baseline (Tier 1)
```bash
python Optimization/run_scripts/tier1_legacy_baseline.py
```
*   **Output**: `Optimization/outputs/tier1_legacy_baseline/`

### Running the JAX Optimizer (Tier 3)
```bash
python Optimization/run_scripts/tier3_global_jax.py
```
*   **Output**: `Optimization/outputs/global_jax/`

## Troubleshooting
*   **Platform Support**: Scripts are OS-agnostic (Mac, Windows, Linux).
*   **Cores**: By default, scripts utilize all available CPU cores.
