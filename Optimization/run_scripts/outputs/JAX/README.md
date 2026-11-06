# JAX Optimization Outputs and Script Guide

This directory (`Optimization/run_scripts/outputs/JAX`) contains the results from various JAX-based QKD optimization scripts.

## 📂 Understanding the Folder Names

Each output folder name tells you exactly which script and strategy produced it.

| 🏷️ Folder Prefix | 📜 Source Script (in `../JAX_variants/`) | 🧠 Strategy Description |
| :--- | :--- | :--- |
| **`run_BASELINE_...`** | `BB84_Parameters_2014_Optimization_Jax_updated.py` | **Standard Optimization**: Uses L-BFGS-B across all distances. Good for general trends but may get stuck in local minima at long distances (>100km). |
| **`run_HYBRID_...`** | `BB84_HighPerformance_LongDistance.py` | **Hybrid Approach**: Uses L-BFGS-B for short distances (<150km) and switches to **Dual Annealing** for long distances to escape local minima. Best balance of speed and accuracy. |
| **`run_PURE_DA_...`** | `BB84_PureDualAnnealing.py` | **Pure Dual Annealing**: Forces global optimization (Dual Annealing) for all distances >100km. Very slow but extremely robust against local minima. |
| **`run_ULTRA_...`** | `BB84_UltraPrecision.py` | **Ultra Precision**: Uses massive initial exploration (100 candidates) followed by high-precision polishing (`ftol=1e-15`). Designed for finding the absolute theoretical maximum key rate. |
| **`run_HESSIAN_...`** | `BB84_Hessian_Optimization.py` | **Hessian Optimization**: Uses **Trust-Region Newton-CG** with exact Hessian from JAX. Robust for non-convex landscapes and saddle points. |

---

## 📊 How to Analyze Results

Each run folder contains:
1.  **`qkd_optimization_results_*.json`**: Raw data of every optimization attempt.
2.  **`qkd_grouped_dataset_*.json`**: Processed data sorted by fiber length, ideal for plotting.
3.  **`detailed_plots/`**: PNG images visualizing the key rate and parameters for specific `n_X` values.

### Comparison
To compare results between different strategies (e.g., Hybrid vs Baseline), you can load the `qkd_grouped_dataset_*.json` files from their respective folders and plot the `key_rate` vs `fiber_length`.

---

## 🛠️ Usage Examples

To run a new optimization, go to the `JAX_variants` directory and run the desired script:

```bash
cd ../JAX_variants

# Run Standard Baseline
python BB84_Parameters_2014_Optimization_Jax_updated.py

# Run Hybrid (Recommended for Long Distance)
python BB84_HighPerformance_LongDistance.py
```

The results will automatically appear in this folder with the correct prefix.
