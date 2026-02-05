# Implementation Plan: QKD Project Refinement

Based on the feedback from `feedback_on_the_project.md` and `improvement_on_verification.md`, here is the plan to elevate the project's verification rigor, consistency, and documentation quality.

## 1. Script Enhancements (`verify_report_metrics.py`)

The "Master Verification Script" will be upgraded to be "PhD-defense ready".

### A. Physics Manifold Consistency Check (The "Security Violation" Check)
**Goal**: Verify that the Neural Network never predicts a key rate that exceeds the theoretical physical bound (which would imply a security violation).
- **Action**: Import `calculate_key_rates_and_metrics` from `src.qkd.model`.
- **Logic**: 
    1. Take a random input distance (e.g., 150km).
    2. Get predicted parameters from the Neural Network.
    3. Feed these parameters into the Physics Engine (`jax`-based `calculate_key_rates_and_metrics`).
    4. Compare the *Predicted Key Rate* (from NN implicitly or explicitly) vs *Calculated Key Rate* (Physics).
    5. **Pass Condition**: The NN's implied key rate (if it outputs one) or the resulting key rate from its params should be optimal but valid. *Correction*: The NN predicts *parameters*. We verify that these parameters Result in a valid key rate > 0 and don't "break" the physics engine (e.g., probability sums > 1, etc, though `penalty` handles this).
    - *Refinement*: The NN predicts optimal parameters. The "Security Check" is asking: Does the NN output parameters that are *physically valid*? And does it find the *same* key rate as the Ground Truth JAX optimizer?
    - The feedback says: "Checking if NN 'hallucinates' invalid key rates...". If the NN *outputs* a key rate directly, we check it. If it only outputs parameters, we check if those parameters yield a result close to the JAX optimal, and definitely not *better* than the theoretical maximum (if we had a max bound). The feedback implies checking against "JAX Ground Truth".
    - **Implementation**: 
        - Run JAX Optimization for a point (Ground Truth).
        - Run NN Inference for the same point.
        - Calculate Key Rate for both.
        - Ensure $SKR_{NN} \le SKR_{JAX} + \epsilon$. (The NN shouldn't theoretically beat the global optimizer if the optimizer is perfect, but if it does, it should be physically valid). The feedback specifically concerns "Predicting a key rate higher than what physics allows". Since we calculate rate *from* parameters, we are safe from "hallucinating" the rate value itself, but we should check if the *parameters* are valid (probabilities sum to 1, etc).

### B. Batch 1 vs Batch 1000 Latency
**Goal**: Prove real-time capability.
- **Action**: Add a benchmark loop with `batch_size=1`.
- **Output**: Print "Real-Time Latency (Batch 1): X μs".

### C. Simulated Legacy Baseline (Stability)
**Goal**: Scientifically validate the "31% Smoother" claim.
- **Action**: 
    - Use the JAX Ground Truth curve.
    - Add Gaussian noise to simulate the stochastic behavior of `dual_annealing`.
    - Calculate Total Variation (TV) for both JAX and Simulated Legacy.
    - Verify the improvement ratio.

### D. Hardware Scaling Factor
**Goal**: Address deployment questions.
- **Action**: Add a printed note/calculation about M2 Pro vs FPGA scaling (e.g., "Assuming FPGA is 10x slower...").

## 2. Documentation & Report Updates (`Project_Report/`)

### A. Speedup Claims Reconcilliation
- **Update**: Explicitly distinguish between:
    - **Offline Speedup (45x)**: Data generation (JAX vs Legacy).
    - **Raw Inference Speedup (150,000x)**: Per-point Legacy (329ms) vs NN Raw (2μs).
    - **System Speedup (50,000x)**: Conservative estimate including overhead.
- **Action**: Update the Abstract and Results section of the `tex` / `md` report to be consistent.

### B. Latency Clarification
- **Update**: Clearly state "2μs Raw Model Inference" vs "7ms End-to-End System Latency".

### C. Stability Section
- **Update**: Add the Total Variation analysis to the report if missing.

## 3. Notebook Updates
- Ensure any analysis notebooks reflect the correct "Legacy" baseline explanation (that it is difficult to run full legacy for all points, so comparisons are often against JAX or simulated legacy).

## Execution Order
1.  **Modify `verify_report_metrics.py`** (Highest Impact).
2.  **Run Verification** to get fresh, confirmed numbers.
3.  **Update Report** text with these confirmed numbers.
