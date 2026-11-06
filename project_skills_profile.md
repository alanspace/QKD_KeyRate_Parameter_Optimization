# Project Profile: Quantum Key Distribution Network Optimization

This profile highlights the comprehensive knowledge, technical skills, and tools employed in the **QKD KeyRate Parameter Optimization** project. It serves as a showcase of capabilities across quantum physics modeling, advanced machine learning, and high-performance scientific computing.

## 🔬 Domain Knowledge & Physics Modeling
- **Quantum Cryptography:** In-depth understanding of Quantum Key Distribution (QKD) systems, specifically the finite-key decoy-state BB84 protocol.
- **Analytical Physics Modeling:** Implementation of rigorous security bounds for practical decoy-state QKD (based on C. C. W. Lim et al., 2014).
- **Secret Key Rate (SKR) Analysis:** Calculating and optimizing SKR using simulated physical parameters to ensure proper exponential decay and secure communication distance extension.
- **Physical Parameter Optimization:** Formulating and solving the optimization of operational parameters (signal/decoy intensities $\mu_k$, probabilities $P_{\mu_k}$, basis choice $P_x$) to maximize the key rate.

## 🤖 Machine Learning & Artificial Intelligence
- **Deep Learning Frameworks:** Proficiency in **PyTorch** for designing and training deep neural networks.
- **Surrogate Modeling:** Utilizing Neural Networks as high-speed surrogate models to predict complex, computationally expensive physics optimization tasks.
- **Model Evaluation & Benchmarking:** Rigorous evaluation of model accuracy, tracking relative predictive error against numerically optimized ground truth data.
- **Future-Forward Architecture:** Planning for the integration of **Physics-Informed Neural Networks (PINNs)** to embed rate equations directly into the ML loss function.

## ⚡ High-Performance Numerical Optimization
- **JAX-Accelerated Computing:** Leveraging **JAX** (`jax.grad`, JIT compilation) for state-of-the-art, gradient-based optimization, replacing slow global search algorithms.
- **Algorithmic Expertise:** Implementation, comparison, and tuning of diverse solver algorithms:
    - **Global Search:** Pure NumPy/SciPy `dual_annealing` for baseline verification.
    - **Adaptive Gradient Descent:** JAX-accelerated Adaptive L-BFGS-B (producing smooth, physically realistic parameter curves).
    - **Segmented & Parallel Execution:** Strategies for splitting execution spaces to accelerate simulation.
- **Computational Speedup:** Achieving a $>6,000\times$ practical end-to-End latency speedup and $45\times$ faster background data generation through modern solvers.

## 💻 Software Engineering & MLOps
- **Languages:** Advanced **Python 3.10+** for scientific computing.
- **Interactive Dashboards:** Deployment of real-time AI optimization tools via **Streamlit Cloud** for interactive, live data visualization.
- **Automated Workflows (CI/CD):** Experience with GitHub Actions (`ci.yml`) to ensure continuous integration, testing, and build stability.
- **Reproducible Environments:** Managing dependencies and execution environments via `conda` and `requirements.txt`.
- **Hardware Acceleration:** Handling cross-platform JAX installations targeting CPU, Apple Silicon, and NVIDIA GPU (`cuda12`) backends.

## 📊 Data Analysis & Visualization
- **Data Engineering:** Architecting pipelines to generate large-scale, high-quality numerical datasets for NN training.
- **Performance Analysis:** Designing and executing comprehensive benchmark scripts (`benchmark_legacy_vs_nn.py`, `benchmark_speedup.py`).
- **Scientific Visualization:** Crafting high-fidelity plots (e.g., parameter relative errors, optimization stability comparisons) to communicate complex quantitative results effectively.

## 🎓 Academic & Scientific Research
- **Rigorous Problem Solving:** Capability to conduct Master's-level research (KTH Royal Institute of Technology), from initial theory to empirical validation.
- **Verification vs. Reality:** Implementing strict verification protocols to prove that Neural Network outputs align with fundamental physics models without hallucinations.
- **Translational Research:** Bridging theoretical quantum security models with practical, real-time edge deployment capabilities.

---
*Generated based on the technical repositories and implementation details of the ML for QKD Network Optimization project.*
