# Machine Learning for Quantum Key Distribution Network Optimization

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)
![Framework: JAX](https://img.shields.io/badge/Framework-JAX-blueviolet.svg)

This repository contains the code and analysis for the project "Machine Learning for Quantum Key Distribution Network Optimization," which investigates the use of neural networks (NNs) to accelerate the parameter optimization of decoy-state BB84 QKD systems.

## Abstract

Optimizing parameters is crucial for maximizing the performance of Quantum Key Distribution (QKD) systems, but traditional numerical methods are computationally prohibitive for real-time applications, especially on resource-constrained platforms like drones or single-board computers. This study investigates the efficacy of neural networks (NNs) as a high-speed alternative to Dual Annealing (DA) for determining optimal operational parameters (signal/decoy intensities `μk`, probabilities `Pμk`, basis choice `Px`) for the finite-key decoy-state BB84 protocol. We demonstrate that a trained NN can predict near-optimal parameters with high accuracy, achieving a **~270-fold speedup** compared to conventional optimization, making sophisticated QKD optimization practical for dynamic scenarios and low-power devices.

## The Problem: The Optimization Bottleneck

Practical QKD systems require careful tuning of operational parameters to maximize the secure key rate (SKR) under varying channel conditions (e.g., changing fiber length, atmospheric turbulence).

- **Traditional Methods are Slow:** Numerical optimization algorithms like Dual Annealing are effective but computationally intensive. Finding the optimal parameters for a single operating point can take minutes on a multi-core CPU.
- **Real-Time Adaptation is Infeasible:** This latency makes it impossible to perform on-the-fly parameter adjustments in dynamic environments (e.g., a QKD-equipped drone or satellite) or on devices with limited computational power.

This project validates a machine learning approach to overcome this bottleneck.

## Methodology

The core idea is to use a slow but accurate optimization method (Dual Annealing) to generate a large dataset of "optimal solutions" and then train a fast neural network to approximate this optimization process instantly.

### 1. Data Generation (The "Ground Truth")

- A comprehensive QKD simulation based on the finite-key decoy-state BB84 protocol (as described in Lim et al., 2014) was implemented in **JAX** for high-performance, differentiable calculations.
- The **Dual Annealing** algorithm from SciPy was used to perform a global search for the optimal parameters (`μ1`, `μ2`, `Pμ1`, `Pμ2`, `Px`) that maximize the SKR.
- This optimization was run for **6,000 different scenarios**, covering a wide range of fiber lengths (0-200 km) and post-processing block sizes (`nx` from 10⁴ to 10⁹).
- The resulting dataset maps experimental conditions to their corresponding optimal parameters and maximum SKR.

### 2. Neural Network Training

- A **PyTorch**-based feed-forward neural network was designed to learn the mapping from experimental conditions to optimal parameters.
- **Architecture:**
  - **Input Layer:** 4 neurons (normalized `L`, `Pdc`, `ed`, `nx`).
  - **Hidden Layers:** 3 fully-connected layers (16, 32, 16 neurons) with ReLU activation.
  - **Output Layer:** 5 neurons (predicting `μ1`, `μ2`, `Pμ1`, `Pμ2`, `Px`) with a linear activation.
- **Training:** The model was trained for 5,000 epochs using the Adam optimizer, Mean Squared Error (MSE) loss, and a `ReduceLROnPlateau` learning rate scheduler. Training was accelerated using the GPU (Apple Silicon MPS backend).

## Key Results

The trained neural network provides a powerful combination of speed and accuracy.

- **🚀 Massive Speedup:** NN inference for 100 operating points takes **~1 second**, whereas the original Dual Annealing optimization requires an estimated **4.5 minutes** for the same task. This represents a **~270x speedup**.

- **🎯 High Accuracy:** The NN predictions closely match the numerically optimized ground truth.
  - The predicted Secret Key Rate (SKR) shows excellent agreement with the optimized SKR across all trained block sizes.
  - For an unseen intermediate block size (`nx = 5 × 10⁸`), the relative error in the final SKR remained within an acceptable **±5%**, even in the challenging high-loss regime near the transmission limit.

- **💡 Excellent Generalization:** The network successfully learned the underlying physics, allowing it to accurately interpolate and predict optimal parameters for conditions it was not explicitly trained on.

<p align="center">
  <img src="https://github.com/alanspace/QKD_KeyRate_Parameter_Optimization/blob/main/NeuralNetwork/image/keyrate_parameters_5e8.png" alt="Predicted vs Optimized Key Rates" width="80%">
  <br>
  <em>Figure: Comparison of SKR from numerically optimized parameters (solid lines) vs. NN-predicted parameters (markers) for an unseen test case (nx = 5x10⁸). The near-perfect overlap demonstrates the model's high accuracy and generalization.</em>
</p>


## Getting Started

### Prerequisites

- Python 3.9 or higher
- GPU support for PyTorch is highly recommended for training (e.g., NVIDIA with CUDA or Apple Silicon with MPS).

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/QKD-ML-Optimization.git
    cd QKD-ML-Optimization
    ```

2.  **Create a virtual environment and activate it:**
    ```bash
    python -m venv qkd
    source qkd/bin/activate  # On Windows, use `qkd\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: Installing JAX and PyTorch with specific hardware acceleration (CUDA/MPS) might require separate commands. Please refer to their official documentation.*

### Usage

1.  **Explore the Data (Optional):** Open `notebooks/1_Data_Generation_and_Analysis.ipynb` to see how the ground-truth dataset was generated and analyzed. Running the optimization from scratch is time-consuming. A pre-generated dataset is available in the `data/` directory.

2.  **Train the Model (Optional):** Run `notebooks/2_Neural_Network_Training.ipynb` to train the neural network on the dataset. This will save the trained model weights to the `models/` directory.

3.  **Perform Fast Inference:** The main application is to use the pre-trained model for instant predictions. Open `notebooks/3_Evaluation_and_Inference.ipynb` to see how to:
    - Load the trained model and scalers.
    - Provide new experimental conditions (fiber length, block size, etc.).
    - Get the predicted optimal QKD parameters in milliseconds.
    - Evaluate the performance of the predicted parameters.

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


This work is based on the analytical model presented in:

Lim, C. C. W., et al. (2014). "Concise security bounds for practical decoy-state quantum key distribution". Physical Review A, 89(2), 022307.
License
This project is licensed under the MIT License - see the LICENSE file for details.
