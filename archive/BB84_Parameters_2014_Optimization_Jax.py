# %% [markdown]
# # BB84 QKD Parameters Optimization

# %% [markdown]
# ## Fiber Lengths and n_X Values
# Fiber lengths are created from 0 to 200 km in 0.1 km steps, resulting in 2001 unique fiber lengths. \
# $n_X$ values are used ranging from $10^6$ to $10^{10}$, creating 5 unique values. \
# Form all combinations of fiber lengths and $n_X$, resulting in 2001 $\times$ 5 = 10,005 combinations, which aligns with the aim to generate a large dataset.
# 
# ## Optimization Process:
# For each combination of fiber length $L$ and $n_X$, The parameters $\vec{p}$ = $ [ \mu_1, \mu_2, P_{\mu_1}, P_{\mu_2}, P_X ]$ are optimized using dual_annealing, which is a global optimization algorithm. \
# The objective function is wrapped to evaluate the key rate for a specific combination of fiber length and $n_X$. 
# 
# ## Parallelization:
# joblib’s Parallel is used to run the optimization for all combinations in parallel, with 12 threads, making the process efficient. \
# tqdm-joblib is also used to track progress visually. 
# 
# ## Dataset Creation:
# The results of the optimization ($e_1, e_2, e_3, e_4, n_X, R, p_{opt}$) are collected into a dataset. \
# This dataset is saved to a file (training_dataset.json) for training a neural network. 
# 
# 
# 

# %% [markdown]
# ## Setup

# %% [markdown]
# ### Framework Selection:JAX
# Functional Programming: Simplify scientific computation and optimization workflows. \
# Smaller Footprint: Ideal if the project doesn’t leverage TensorFlow’s broader ecosystem.
# 
# ##### Key Considerations
# For numerically intensive workloads (e.g., optimization tasks like QKD key rate calculations), JAX excels due to its lightweight functional paradigm.
# 
# ##### Conclusion
# Use JAX for performance-critical, purely numerical optimization tasks with minimal dependencies on machine learning frameworks.

# %% [markdown]
# ## Imports

# %%
# Import necessary libraries
import os
import time
import json
import functools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
from concurrent.futures import ProcessPoolExecutor

# JAX imports
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.scipy.special import logsumexp, gamma
from jax.experimental import pjit
from jax.sharding import Mesh

# SciPy imports
from scipy.optimize import minimize, dual_annealing, differential_evolution, Bounds
from math import exp, factorial

# JAX configuration for double precision
jax.config.update("jax_enable_x64", True)

# %%
print(jax.devices())

# %% [markdown]
# ## Experimental Parameters

# %%
# e_1
# Fiber lengths
Ls = jnp.linspace(5, 200, 1000)  # Fiber lengths in km
L_BC = Ls
e_1 = L_BC / 100

#e_2
P_dc_value = 6*10**-7  # Dark count probability
Y_0 = P_dc_value
# 2.7*10** -7
# P_dc = 6 * 10 ** (-7)   # given in the paper, discussed with range from 10^-8 to 10^-5
# e_2 = -jnp.log(Y_0)

# e_3
e_mis = 5 * 10 ** -3  # Misalignment error probability
# 0.026 
e_d = e_mis
e_3 = e_d * 100
e_mis = 5*1e-3 # given in the paper, discussed with range from 0 to 0.1 

# e_4
# Detected events
n_X_values = [10 ** s for s in range(6, 11)]  # Detected events
# n_X_values = jnp.array([10**s for s in range(6, 11)], dtype=jnp.int64)
N = n_X_values
e_4 = N

# %% [markdown]
# ## Other Parameters

# %%
alpha = 0.2  # Attenuation coefficient (dB/km), given in the paper
eta_Bob = 0.1  # Detector efficiency, given in the paper
P_ap = 0  # After-pulse probability
f_EC = 1.16  # Error correction efficiency
# secutity error 
epsilon_sec = 1e-10 # is equal to kappa * secrecy length Kl, range around 1e-10 Scalar, as it is a single value throughout the calculations.
# correlation error
epsilon_cor = 1e-15 # given in the paper, discussed with range from 0 to 10e-10
# Dark count probability
n_event = 1  # for single photon event
# Misalignment error probability
# 4*1e-2          # given in the paper, discussed with range from 0 to 0.1
kappa = 1e-15           # given in the paper
f_EC = 1.16             # given in the paper, range around 1.1


# %% [markdown]
# ## Optimal Paramters

# %%
# p, optimal parameters
# p_1 = mu_1 = 6e-1
# p_2 = mu_2 = 2e-1
# mu_3 = 2e-4
# mu_k_values = [mu_1, mu_2, mu_3]
# p_3 = P_mu_1 = 0.65
# p_4 = P_mu_2 = 0.3
# P_mu_3 = 1 - P_mu_1 - P_mu_2
# p_mu_k_values = [P_mu_1, P_mu_2, P_mu_3]
# p_5 = P_X_value = 5e-3
# P_Z_value = 1 - P_X_value
def optimal_parameters(params):
    mu_1, mu_2, P_mu_1, P_mu_2, P_X_value = params
    mu_3 = 2e-4
    P_mu_3 = 1 - P_mu_1 - P_mu_2
    P_Z_value = 1 - P_X_value
    mu_k_values = jnp.array([mu_1, mu_2, mu_3])
    return params, mu_3, P_mu_3, P_Z_value, mu_k_values

# %% [markdown]
# ## Functions
# 
# The calculate_factorial function provided uses the gamma function to compute the factorial of a number  n . This is mathematically correct because the gamma function  \Gamma(n+1)  is equivalent to the factorial  n!  for non-negative integers  n .
# 
# ## Mathematical Background
# The gamma function is defined as:
# $\Gamma(x) = \int_0^\infty t^{x-1} e^{-t} \, dt$ \
# For positive integers, the gamma function satisfies the relationship: \
# $\Gamma(n + 1) = n!$
# \
# JAX does not have a built-in factorial function, but it does support the gamma function. This makes the approach valid and compatible with JAX for automatic differentiation and JIT compilation.

# %%
from QKD_Functions import (
    calculate_factorial,
    calculate_tau_n,
    calculate_eta_ch,
    calculate_eta_sys,
    calculate_D_mu_k,
    calculate_n_X_total,
    calculate_N,
    calculate_n_Z_total,
    calculate_e_mu_k,
    calculate_e_obs,
    calculate_h,
    calculate_lambda_EC,
    calculate_sqrt_term,
    calculate_tau_n,
    calculate_n_pm, 
    calculate_S_0,
    calculate_S_1,
    calculate_m_mu_k,
    calculate_m_pm,
    calculate_v_1,
    calculate_gamma,
    calculate_Phi,
    calculate_LastTwoTerm,
    calculate_l,
    calculate_R,
    experimental_parameters,
    other_parameters,
    calculate_key_rates_and_metrics,
    penalty, 
    objective,
    objective_with_logging
)

# %%
# Bounds for optimization parameters
bounds = [
    (1*1e-12, 1),  # mu_1, if it represents a mean photon number, adjust max as needed
    (1*1e-12, 1),  # mu_2
    (1*1e-12, 1),   # P_mu_1, probability
    (1*1e-12, 1),   # P_mu_2, probability
    (1*1e-12, 1),   # P_X_value, probability
]

# %%
def optimize_single_instance(input_params, bounds, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event, max_retries=5):
    """
    Optimize key rates for a given fiber length and n_X value using dual annealing 
    and Nelder-Mead optimization, ensuring parameters remain within bounds and retry
    if log10(key_rate) > 0.
    """
    L, n_X = input_params

    def wrapped_objective(params):
        # Negative because we minimize, but key rates need maximization
        return -objective(params, L, n_X, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event)[0]

    best_key_rate = float('-inf')
    best_params = None

    for attempt in range(max_retries):
        try:
            # Step 1: Perform global optimization using dual annealing
            global_result = dual_annealing(func=wrapped_objective, bounds=bounds)

            # Step 2: Refine results using local optimization (Nelder-Mead)
            local_result = minimize(
                fun=wrapped_objective,
                x0=global_result.x,  # Start from global optimization result
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6}  # Tighten tolerances
            )

            # Extract the optimized parameters and key rate
            optimized_params = local_result.x
            optimized_key_rate = -local_result.fun  # Convert back to key rate (positive)

            # Validate the optimized parameters against the bounds
            out_of_bounds = any(
                param < b[0] or param > b[1]
                for param, b in zip(optimized_params, bounds)
            )

            # Retry if key rate is abnormal (log10(key_rate) > 0)
            if np.log10(max(optimized_key_rate, 1e-10)) > 0:
                print(f"Attempt {attempt + 1}: Abnormal key rate detected (log10 > 0). Retrying...")
                continue

            if not out_of_bounds:
                # If all parameters are valid, update best results if key rate is improved
                if optimized_key_rate > best_key_rate:
                    best_key_rate = optimized_key_rate
                    best_params = optimized_params

                # Stop retrying if parameters stabilize
                if best_params is not None and np.allclose(best_params, optimized_params, atol=1e-4):
                    break
            else:
                print(f"Attempt {attempt + 1}: Parameters out of bounds. Retrying...")

        except Exception as e:
            print(f"Attempt {attempt + 1}: Optimization error: {e}. Retrying...")

    if best_params is not None:
        return L, n_X, best_key_rate, best_params

    # If we exhaust retries, raise an error or return NaN results
    print(f"Optimization failed after {max_retries} retries.")
    return L, n_X, float('nan'), [float('nan')] * len(bounds)

# %%
# Define input parameters for a single instance
single_input = (Ls[0], n_X_values[0])  # Example: first fiber length and first n_X value

# Measure start time
start_time = time.time()

# Run the optimization for the single instance
L, n_X, optimized_key_rate, optimized_params = optimize_single_instance(
    single_input, bounds, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event
)

# Measure end time
end_time = time.time()

# Output the results with parameter names
parameter_names = ["mu_1", "mu_2", "P_mu_1", "P_mu_2", "P_X_value"]
optimized_parameters = {name: value for name, value in zip(parameter_names, optimized_params)}

print(f"Optimization for a single instance took {end_time - start_time:.2f} seconds.")
print(f"Fiber Length: {L} km, Detected Events (n_X): {n_X}")
print(f"Optimized Key Rate: {optimized_key_rate:.3e}")
print("Optimized Parameters:")
for name, value in optimized_parameters.items():
    print(f"  {name}: {value:.6f}")

# %% [markdown]
# Reasonableness Check for Execution Time:
# 
# A single optimization took 1.81 seconds. For 10,000 instances, the time would scale proportionally if no optimizations (e.g., batch parallelism) are applied:
# 
# $ \text{Total Time} = 1.81 \times 10,000 \approx 5.03 \, \text{hours}$ 
# 
# With 12 CPU cores using joblib, the time should reduce by approximately  1 / 12 :
# 
# $ \text{Total Time with 12 CPUs} \approx 5.03 / 12 \approx 25.13 \, \text{minutes}$
# 

# %% [markdown]
# ## Parallel Dataset Generation Using joblib

# %%
#     return dataset
def generate_dataset(Ls, n_X_values, bounds, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event):
    """
    Generate a dataset by optimizing key rates for various fiber lengths and n_X values.
    """
    inputs = [(L, n_X) for L in Ls for n_X in n_X_values]

    print("Generating dataset...")
    results = []
    with tqdm(total=len(inputs), desc="Generating Dataset") as progress_bar:
        results = Parallel(n_jobs=12)(
            delayed(optimize_single_instance)(
                input_params, bounds, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event
            )
            for input_params in inputs
        )
        progress_bar.update(len(results))

    # Process results into a dataset
    dataset = []
    for L, n_X, penalized_key_rate, optimized_params in results:
        # Skip invalid key rates
        if penalized_key_rate <= 0:
            print(f"Skipping invalid key rate: {penalized_key_rate}")
            continue

        # Extract optimized parameters (ensure the order matches the bounds setup)
        mu_1, mu_2, P_mu_1, P_mu_2, P_X_value = optimized_params

        # Compute normalized parameters
        e_1 = L / 100  # Normalize fiber length
        e_2 = -jnp.log10(P_dc_value)  # Normalize dark count probability
        e_3 = e_mis * 100  # Normalize misalignment error probability
        e_4 = jnp.log10(n_X)  # Normalize number of pulses

        # Append processed data
        dataset.append({
            "e_1": e_1,
            "e_2": e_2,
            "e_3": e_3,
            "e_4": e_4,
            "key_rate": max(penalized_key_rate, 1e-10),  # Enforce non-negative key rate
            "optimized_params": {
                "mu_1": mu_1,
                "mu_2": mu_2,
                "P_mu_1": P_mu_1,
                "P_mu_2": P_mu_2,
                "P_X_value": P_X_value,
            }
        })

    return dataset

# %%


# %%
def convert_to_serializable(obj):
    """
    Recursively convert JAX arrays and other non-JSON serializable objects
    to standard Python types.
    """
    if isinstance(obj, jnp.ndarray):
        return obj.tolist()  # Convert JAX array to a Python list
    elif isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert NumPy array to a Python list
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    else:
        return obj  # Return other types as-is

# %%
Ls = jnp.linspace(1, 200, 100)  # Fiber lengths (5 km to 200 km)
n_X_values = jnp.logspace(6, 11, 100)  # Detected events (10^6 to 10^10)

print(f"L ranges from {min(Ls):.2f} to {max(Ls):.2f} km, with {len(Ls)} points.")
print(f"n_X_values ranges from {min(n_X_values):.2e} to {max(n_X_values):.2e}, with {len(n_X_values)} points.")
print(f"total {len(Ls) * len(n_X_values)} points, the difference is {len(Ls) * len(n_X_values)- 41*10000}")

# %%
import time

# Example input parameters
example_input = (Ls[0], n_X_values[0])

start_time = time.time()
# Run the optimization for a single instance
optimize_single_instance(
    example_input, bounds, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event
)
end_time = time.time()

# Measure and assign the runtime
single_instance_time = end_time - start_time
print(f"Time per single instance: {single_instance_time:.2f} seconds")

total_points = 10000
parallel_jobs = 12
time_per_instance = single_instance_time  # e.g., 2 seconds

# Total instances divided by parallel jobs
time_estimate = (total_points / parallel_jobs) * time_per_instance

print(f"Estimated time to generate {total_points} entries: {time_estimate / 3600:.2f} hours")

# %% [markdown]
# 
# np.linspace(0.1, 250, 10): \
# Generates 10 equally spaced values between 0.1 and 250. For example: [0.1, 27.88, 55.66, ..., 250].
# Represents 10 fiber lengths to evaluate, rather than 1000.
# np.logspace(6, 8, 5): \
# Generates 5 logarithmically spaced values between 10^6 and 10^8. For example: [1e6, 3.16e6, 1e7, ..., 1e8].
# Represents the number of detected events (n_X) in a smaller range with fewer steps.

# %%
# Define parameter space
Ls = jnp.linspace(1, 200, 100)  # Fiber lengths (5 km to 200 km)
n_X_values = jnp.logspace(6, 10, 1000)  # Detected events (10^6 to 10^10)

# Profile a single optimization instance (optional debugging)
import cProfile
cProfile.run("optimize_single_instance((Ls[0], n_X_values[0]), bounds, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event)")

# Measure total dataset generation time
import time
start_time = time.time()

# Step 1: Generate the raw dataset
raw_dataset = generate_dataset(
    Ls, n_X_values, bounds, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event
)

# Step 2: Filter out entries with non-positive key rates
# Filter out entries with negative key rates
filtered_dataset = [entry for entry in raw_dataset if entry["key_rate"] > 0]

end_time = time.time()

# Step 3: Convert the filtered dataset to a JSON-serializable format
serializable_dataset = convert_to_serializable(filtered_dataset)

# Step 4: Save the filtered dataset to a JSON file
output_filename = "total_training_dataset.json"
with open(output_filename, "w") as f:
    json.dump(serializable_dataset, f)

print(f"Filtered dataset saved as '{output_filename}'.")

# %%
print(f"Raw dataset size: {len(raw_dataset)}")
key_rates = [entry["key_rate"] for entry in raw_dataset]
print(f"Number of negative key rates: {sum(kr <= 0 for kr in key_rates)}")
print(f"Number of positive key rates: {sum(kr > 0 for kr in key_rates)}")

# %%
# import jax.numpy as jnp
# import json
# import time

# # Function to dynamically expand parameter space
# def expand_parameter_space(Ls, n_X_values, scale_factor=1.1):
#     """
#     Expand the parameter space by increasing resolution or range.
#     scale_factor: Multiplier for expanding range or density.
#     """
#     new_Ls = jnp.linspace(Ls[0], Ls[-1] * scale_factor, len(Ls) * 2)
#     new_n_X_values = jnp.logspace(
#         jnp.log10(n_X_values[0]),
#         jnp.log10(n_X_values[-1] * scale_factor),
#         len(n_X_values) * 2,
#     )
#     return new_Ls, new_n_X_values

# # Target size for the dataset
# target_size = *100*5

# # Initial parameter space
# Ls = jnp.linspace(5, 200, 100)
# n_X_values = jnp.logspace(6, 10, 11)

# # # Initialize bounds and other constants here (example placeholders)
# # bounds = {...}  # Replace with actual bounds
# # alpha = ...
# # eta_Bob = ...
# # P_dc_value = ...
# # epsilon_sec = ...
# # epsilon_cor = ...
# # f_EC = ...
# # e_mis = ...
# # P_ap = ...
# # n_event = ...

# # Start timing
# start_time = time.time()

# # Regeneration loop
# filtered_dataset = []
# while len(filtered_dataset) < target_size:
#     # Step 1: Generate the raw dataset
#     raw_dataset = generate_dataset(
#         Ls, n_X_values, bounds, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event
#     )
    
#     # Step 2: Filter entries with non-positive key rates
#     filtered_dataset = [entry for entry in raw_dataset if entry["key_rate"] > 0]
    
#     # Check if target size is met
#     if len(filtered_dataset) < target_size:
#         print(f"Dataset size {len(filtered_dataset)} is smaller than target {target_size}. Expanding parameter space...")
#         Ls, n_X_values = expand_parameter_space(Ls, n_X_values)

# # Truncate dataset if it exceeds target size
# if len(filtered_dataset) > target_size:
#     filtered_dataset = filtered_dataset[:target_size]

# # Convert dataset to JSON-serializable format
# serializable_dataset = convert_to_serializable(filtered_dataset)

# # Save the dataset to a JSON file
# output_filename = "filtered_training_dataset.json"
# with open(output_filename, "w") as f:
#     json.dump(serializable_dataset, f)

# end_time = time.time()

# print(f"Filtered dataset saved as '{output_filename}'.")
# print(f"Final dataset size: {len(filtered_dataset)}")
# print(f"Time taken: {end_time - start_time:.2f} seconds")

# %%
# Check the number of entries in the filtered dataset
print(f"Number of valid entries: {len(filtered_dataset)}")

# %%
# Load the dataset
with open("training_dataset.json", "r") as f:
    data = json.load(f)

# Extract fiber lengths and key rates
e_1 = jnp.array([item["e_1"] * 100 for item in data])  # Denormalize fiber lengths (convert to km)
key_rate = jnp.array([item["key_rate"] for item in data])  # Correct key name

# Extract optimized parameters
mu_1 = jnp.array([item["optimized_params"]["mu_1"] for item in data])  # Access nested keys
mu_2 = jnp.array([item["optimized_params"]["mu_2"] for item in data])
P_mu_1 = jnp.array([item["optimized_params"]["P_mu_1"] for item in data])
P_mu_2 = jnp.array([item["optimized_params"]["P_mu_2"] for item in data])
P_X_value = jnp.array([item["optimized_params"]["P_X_value"] for item in data])

# Sort by fiber length for smooth plotting
sorted_indices = jnp.argsort(e_1)
e_1_sorted = e_1[sorted_indices]
key_rate_sorted = key_rate[sorted_indices]
mu_1_sorted = mu_1[sorted_indices]
mu_2_sorted = mu_2[sorted_indices]
P_mu_1_sorted = P_mu_1[sorted_indices]
P_mu_2_sorted = P_mu_2[sorted_indices]
P_X_value_sorted = P_X_value[sorted_indices]

# Plot the data
plt.figure(figsize=(15, 6))

# Left plot: Penalized Key Rate
plt.subplot(1, 2, 1)
plt.plot(e_1_sorted, jnp.log10(jnp.clip(key_rate_sorted, a_min=1e-10, a_max=None)), label="Penalized Key Rate (log10)")
plt.xlabel("Fiber Length (km)")
plt.ylabel("log10(Penalized Key Rate)")
plt.title("Penalized Key Rate vs Fiber Length")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend()

# Right plot: Optimized Parameters
plt.subplot(1, 2, 2)
plt.plot(e_1_sorted, mu_1_sorted, label="mu_1")
plt.plot(e_1_sorted, mu_2_sorted, label="mu_2")
plt.plot(e_1_sorted, P_mu_1_sorted, label="P_mu_1")
plt.plot(e_1_sorted, P_mu_2_sorted, label="P_mu_2")
plt.plot(e_1_sorted, P_X_value_sorted, label="P_X_value")
plt.xlabel("Fiber Length (km)")
plt.ylabel("Optimized Parameters")
plt.title("Optimized Parameters vs Fiber Length")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend()
# Save the figure
plt.tight_layout()
plt.savefig("optimized_parameters_plot.png", dpi=300, bbox_inches="tight")  # Save with high resolution
plt.show()

# %%
# Define parameter space
Ls = jnp.linspace(5, 200, 100)  # Fiber lengths
n_X_values = jnp.array([1e10])  # Single value for n_X

import cProfile
cProfile.run("optimize_single_instance((Ls[0], n_X_values[0]), bounds, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event)")

# Measure total dataset generation time
import time
start_time = time.time()

dataset = generate_dataset(
    Ls, n_X_values, bounds, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event
)

end_time = time.time()
print(f"Dataset generation completed in {(end_time - start_time) / 60:.2f} minutes.")

# Convert dataset to a JSON-serializable format
serializable_dataset = convert_to_serializable(dataset)

# Save to JSON
output_filename = "training_dataset.json"
with open(output_filename, "w") as f:
    json.dump(serializable_dataset, f)

print(f"Dataset saved as '{output_filename}'.")

# Load the dataset
with open("training_dataset.json", "r") as f:
    data = json.load(f)

# Extract fiber lengths and key rates
e_1 = jnp.array([item["e_1"] * 100 for item in data])  # Denormalize fiber lengths (convert to km)
key_rate = jnp.array([item["key_rate"] for item in data])  # Correct key name

# Extract optimized parameters
mu_1 = jnp.array([item["optimized_params"]["mu_1"] for item in data])  # Access nested keys
mu_2 = jnp.array([item["optimized_params"]["mu_2"] for item in data])
P_mu_1 = jnp.array([item["optimized_params"]["P_mu_1"] for item in data])
P_mu_2 = jnp.array([item["optimized_params"]["P_mu_2"] for item in data])
P_X_value = jnp.array([item["optimized_params"]["P_X_value"] for item in data])

# Sort by fiber length for smooth plotting
sorted_indices = jnp.argsort(e_1)
e_1_sorted = e_1[sorted_indices]
key_rate_sorted = key_rate[sorted_indices]
mu_1_sorted = mu_1[sorted_indices]
mu_2_sorted = mu_2[sorted_indices]
P_mu_1_sorted = P_mu_1[sorted_indices]
P_mu_2_sorted = P_mu_2[sorted_indices]
P_X_value_sorted = P_X_value[sorted_indices]

# Plot the data
plt.figure(figsize=(15, 6))

# Left plot: Penalized Key Rate
plt.subplot(1, 2, 1)
plt.plot(e_1_sorted, jnp.log10(jnp.clip(key_rate_sorted, a_min=1e-10, a_max=None)), label="Penalized Key Rate (log10)")
plt.xlabel("Fiber Length (km)")
plt.ylabel("log10(Penalized Key Rate)")
plt.title("Penalized Key Rate vs Fiber Length")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend()

# Right plot: Optimized Parameters
plt.subplot(1, 2, 2)
plt.plot(e_1_sorted, mu_1_sorted, label="mu_1")
plt.plot(e_1_sorted, mu_2_sorted, label="mu_2")
plt.plot(e_1_sorted, P_mu_1_sorted, label="P_mu_1")
plt.plot(e_1_sorted, P_mu_2_sorted, label="P_mu_2")
plt.plot(e_1_sorted, P_X_value_sorted, label="P_X_value")
plt.xlabel("Fiber Length (km)")
plt.ylabel("Optimized Parameters")
plt.title("Optimized Parameters vs Fiber Length")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend()
# Save the figure
plt.tight_layout()
plt.savefig("optimized_parameters_plot.png", dpi=300, bbox_inches="tight")  # Save with high resolution
plt.show()

# %%
# Define parameter space
Ls = jnp.linspace(5, 200, 100)  # Fiber lengths
n_X_values = jnp.array([1e9])  # Single value for n_X

import cProfile
cProfile.run("optimize_single_instance((Ls[0], n_X_values[0]), bounds, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event)")

# Measure total dataset generation time
import time
start_time = time.time()

dataset = generate_dataset(
    Ls, n_X_values, bounds, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event
)

end_time = time.time()
print(f"Dataset generation completed in {(end_time - start_time) / 60:.2f} minutes.")

# Convert dataset to a JSON-serializable format
serializable_dataset = convert_to_serializable(dataset)

# Save to JSON
output_filename = "training_dataset.json"
with open(output_filename, "w") as f:
    json.dump(serializable_dataset, f)

print(f"Dataset saved as '{output_filename}'.")

# Load the dataset
with open("training_dataset.json", "r") as f:
    data = json.load(f)

# Extract fiber lengths and key rates
e_1 = jnp.array([item["e_1"] * 100 for item in data])  # Denormalize fiber lengths (convert to km)
key_rate = jnp.array([item["key_rate"] for item in data])  # Correct key name

# Extract optimized parameters
mu_1 = jnp.array([item["optimized_params"]["mu_1"] for item in data])  # Access nested keys
mu_2 = jnp.array([item["optimized_params"]["mu_2"] for item in data])
P_mu_1 = jnp.array([item["optimized_params"]["P_mu_1"] for item in data])
P_mu_2 = jnp.array([item["optimized_params"]["P_mu_2"] for item in data])
P_X_value = jnp.array([item["optimized_params"]["P_X_value"] for item in data])

# Sort by fiber length for smooth plotting
sorted_indices = jnp.argsort(e_1)
e_1_sorted = e_1[sorted_indices]
key_rate_sorted = key_rate[sorted_indices]
mu_1_sorted = mu_1[sorted_indices]
mu_2_sorted = mu_2[sorted_indices]
P_mu_1_sorted = P_mu_1[sorted_indices]
P_mu_2_sorted = P_mu_2[sorted_indices]
P_X_value_sorted = P_X_value[sorted_indices]

# Plot the data
plt.figure(figsize=(15, 6))

# Left plot: Penalized Key Rate
plt.subplot(1, 2, 1)
plt.plot(e_1_sorted, jnp.log10(jnp.clip(key_rate_sorted, a_min=1e-10, a_max=None)), label="Penalized Key Rate (log10)")
plt.xlabel("Fiber Length (km)")
plt.ylabel("log10(Penalized Key Rate)")
plt.title("Penalized Key Rate vs Fiber Length")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend()

# Right plot: Optimized Parameters
plt.subplot(1, 2, 2)
plt.plot(e_1_sorted, mu_1_sorted, label="mu_1")
plt.plot(e_1_sorted, mu_2_sorted, label="mu_2")
plt.plot(e_1_sorted, P_mu_1_sorted, label="P_mu_1")
plt.plot(e_1_sorted, P_mu_2_sorted, label="P_mu_2")
plt.plot(e_1_sorted, P_X_value_sorted, label="P_X_value")
plt.xlabel("Fiber Length (km)")
plt.ylabel("Optimized Parameters")
plt.title("Optimized Parameters vs Fiber Length")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend()
# Save the figure
plt.tight_layout()
plt.savefig("optimized_parameters_plot.png", dpi=300, bbox_inches="tight")  # Save with high resolution
plt.show()

# %%
# Define the target value of n_X
n_X_target_log = jnp.log10(1e9)  # Log10 scale of n_X = 10^9

# Filter the dataset for entries matching n_X = 10^10 (based on e_4)
filtered_data = [
    item for item in data if jnp.isclose(item["e_4"], n_X_target_log, atol=1e-4)
]

if not filtered_data:
    print("No entries found matching n_X = 10^9")
else:
    print(f"Found {len(filtered_data)} entries for n_X = 10^9.")

    # Extract fiber lengths and key rates for the filtered data
    e_1_filtered = jnp.array([item["e_1"] * 100 for item in filtered_data])  # Denormalize fiber lengths
    key_rate_filtered = jnp.array([item["key_rate"] for item in filtered_data])

    # Extract optimized parameters
    mu_1_filtered = jnp.array([item["optimized_params"]["mu_1"] for item in filtered_data])
    mu_2_filtered = jnp.array([item["optimized_params"]["mu_2"] for item in filtered_data])
    P_mu_1_filtered = jnp.array([item["optimized_params"]["P_mu_1"] for item in filtered_data])
    P_mu_2_filtered = jnp.array([item["optimized_params"]["P_mu_2"] for item in filtered_data])
    P_X_value_filtered = jnp.array([item["optimized_params"]["P_X_value"] for item in filtered_data])

    # Sort by fiber length for smooth plotting
    sorted_indices_filtered = jnp.argsort(e_1_filtered)
    e_1_sorted_filtered = e_1_filtered[sorted_indices_filtered]
    key_rate_sorted_filtered = key_rate_filtered[sorted_indices_filtered]
    mu_1_sorted_filtered = mu_1_filtered[sorted_indices_filtered]
    mu_2_sorted_filtered = mu_2_filtered[sorted_indices_filtered]
    P_mu_1_sorted_filtered = P_mu_1_filtered[sorted_indices_filtered]
    P_mu_2_sorted_filtered = P_mu_2_filtered[sorted_indices_filtered]
    P_X_value_sorted_filtered = P_X_value_filtered[sorted_indices_filtered]

    # Create the filtered plot
    plt.figure(figsize=(15, 6))

    # Left plot: Penalized Key Rate
    plt.subplot(1, 2, 1)
    plt.plot(
        e_1_sorted_filtered,
        jnp.log10(jnp.clip(key_rate_sorted_filtered, a_min=1e-10, a_max=None)),
        label="Penalized Key Rate (log10)"
    )
    plt.xlabel("Fiber Length (km)")
    plt.ylabel("log10(Penalized Key Rate)")
    plt.title("Penalized Key Rate vs Fiber Length (n_X = 10^10)")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()

    # Right plot: Optimized Parameters
    plt.subplot(1, 2, 2)
    plt.plot(e_1_sorted_filtered, mu_1_sorted_filtered, label="mu_1")
    plt.plot(e_1_sorted_filtered, mu_2_sorted_filtered, label="mu_2")
    plt.plot(e_1_sorted_filtered, P_mu_1_sorted_filtered, label="P_mu_1")
    plt.plot(e_1_sorted_filtered, P_mu_2_sorted_filtered, label="P_mu_2")
    plt.plot(e_1_sorted_filtered, P_X_value_sorted_filtered, label="P_X_value")
    plt.xlabel("Fiber Length (km)")
    plt.ylabel("Optimized Parameters")
    plt.title("Optimized Parameters vs Fiber Length (n_X = 10^10)")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()

    # Save and show the plot
    plt.tight_layout()
    plt.savefig("filtered_optimized_parameters_plot_nX_10_9.png", dpi=300, bbox_inches="tight")
    plt.show()

# %%
# Define the target value of n_X
n_X_target_log = jnp.log10(1e9)  # Log10 scale of n_X = 10^10

# Filter the dataset for entries matching n_X = 10^10 (based on e_4)
filtered_data = [
    item for item in data if jnp.isclose(item["e_4"], n_X_target_log, atol=1e-4)
]

if not filtered_data:
    print("No entries found matching n_X = 10^10.")
else:
    print(f"Found {len(filtered_data)} entries for n_X = 10^10.")

    # Extract fiber lengths and key rates for the filtered data
    e_1_filtered = jnp.array([item["e_1"] * 100 for item in filtered_data])  # Denormalize fiber lengths
    key_rate_filtered = jnp.array([item["key_rate"] for item in filtered_data])

    # Extract optimized parameters
    mu_1_filtered = jnp.array([item["optimized_params"]["mu_1"] for item in filtered_data])
    mu_2_filtered = jnp.array([item["optimized_params"]["mu_2"] for item in filtered_data])
    P_mu_1_filtered = jnp.array([item["optimized_params"]["P_mu_1"] for item in filtered_data])
    P_mu_2_filtered = jnp.array([item["optimized_params"]["P_mu_2"] for item in filtered_data])
    P_X_value_filtered = jnp.array([item["optimized_params"]["P_X_value"] for item in filtered_data])

    # Sort by fiber length for smooth plotting
    sorted_indices_filtered = jnp.argsort(e_1_filtered)
    e_1_sorted_filtered = e_1_filtered[sorted_indices_filtered]
    key_rate_sorted_filtered = key_rate_filtered[sorted_indices_filtered]
    mu_1_sorted_filtered = mu_1_filtered[sorted_indices_filtered]
    mu_2_sorted_filtered = mu_2_filtered[sorted_indices_filtered]
    P_mu_1_sorted_filtered = P_mu_1_filtered[sorted_indices_filtered]
    P_mu_2_sorted_filtered = P_mu_2_filtered[sorted_indices_filtered]
    P_X_value_sorted_filtered = P_X_value_filtered[sorted_indices_filtered]

    # Create the filtered plot
    plt.figure(figsize=(15, 6))

    # Left plot: Penalized Key Rate
    plt.subplot(1, 2, 1)
    plt.plot(
        e_1_sorted_filtered,
        jnp.log10(jnp.clip(key_rate_sorted_filtered, a_min=1e-10, a_max=None)),
        label="Penalized Key Rate (log10)"
    )
    plt.xlabel("Fiber Length (km)")
    plt.ylabel("log10(Penalized Key Rate)")
    plt.title("Penalized Key Rate vs Fiber Length (n_X = 10^10)")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()

    # Right plot: Optimized Parameters
    plt.subplot(1, 2, 2)
    plt.plot(e_1_sorted_filtered, mu_1_sorted_filtered, label="mu_1")
    plt.plot(e_1_sorted_filtered, mu_2_sorted_filtered, label="mu_2")
    plt.plot(e_1_sorted_filtered, P_mu_1_sorted_filtered, label="P_mu_1")
    plt.plot(e_1_sorted_filtered, P_mu_2_sorted_filtered, label="P_mu_2")
    plt.plot(e_1_sorted_filtered, P_X_value_sorted_filtered, label="P_X_value")
    plt.xlabel("Fiber Length (km)")
    plt.ylabel("Optimized Parameters")
    plt.title("Optimized Parameters vs Fiber Length (n_X = 10^10)")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()

    # Save and show the plot
    plt.tight_layout()
    plt.savefig("filtered_optimized_parameters_plot_nX_10_10.png", dpi=300, bbox_inches="tight")
    plt.show()

# %%
Ls = jnp.linspace(5, 200, 100)  # Fiber lengths
n_X_values = jnp.logspace(6, 10, 11)

# Load the dataset
with open("training_dataset.json", "r") as f:
    data = json.load(f)

# Extract fiber lengths and key rates
e_1 = jnp.array([item["e_1"] * 100 for item in data])  # Denormalize fiber lengths (convert to km)
key_rate = jnp.array([item["key_rate"] for item in data])  # Correct key name

# Extract optimized parameters
mu_1 = jnp.array([item["optimized_params"]["mu_1"] for item in data])  # Access nested keys
mu_2 = jnp.array([item["optimized_params"]["mu_2"] for item in data])
P_mu_1 = jnp.array([item["optimized_params"]["P_mu_1"] for item in data])
P_mu_2 = jnp.array([item["optimized_params"]["P_mu_2"] for item in data])
P_X_value = jnp.array([item["optimized_params"]["P_X_value"] for item in data])

# Sort by fiber length for smooth plotting
sorted_indices = jnp.argsort(e_1)
e_1_sorted = e_1[sorted_indices]
key_rate_sorted = key_rate[sorted_indices]
mu_1_sorted = mu_1[sorted_indices]
mu_2_sorted = mu_2[sorted_indices]
P_mu_1_sorted = P_mu_1[sorted_indices]
P_mu_2_sorted = P_mu_2[sorted_indices]
P_X_value_sorted = P_X_value[sorted_indices]

# Plot the data
plt.figure(figsize=(15, 6))

# Left plot: Penalized Key Rate
plt.subplot(1, 2, 1)
plt.plot(e_1_sorted, jnp.log10(jnp.clip(key_rate_sorted, a_min=1e-10, a_max=None)), label="Penalized Key Rate (log10)")
plt.xlabel("Fiber Length (km)")
plt.ylabel("log10(Penalized Key Rate)")
plt.title("Penalized Key Rate vs Fiber Length")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend()

# Right plot: Optimized Parameters
plt.subplot(1, 2, 2)
plt.plot(e_1_sorted, mu_1_sorted, label="mu_1")
plt.plot(e_1_sorted, mu_2_sorted, label="mu_2")
plt.plot(e_1_sorted, P_mu_1_sorted, label="P_mu_1")
plt.plot(e_1_sorted, P_mu_2_sorted, label="P_mu_2")
plt.plot(e_1_sorted, P_X_value_sorted, label="P_X_value")
plt.xlabel("Fiber Length (km)")
plt.ylabel("Optimized Parameters")
plt.title("Optimized Parameters vs Fiber Length")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend()
# Save the figure
plt.tight_layout()
plt.savefig("optimized_parameters_plot.png", dpi=300, bbox_inches="tight")  # Save with high resolution
plt.show()


# %%
# Define the parameters for optimization
L = 0  # Fiber length = 0
n_X = n_X_values[0]  # Use a default n_X value for the calculation

# Run optimization for L = 0
_, _, key_rate_L0, optimized_params_L0 = optimize_single_instance(
    (L, n_X), bounds, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event
)

if key_rate_L0 is not None:
    print(f"Computed Key Rate at Fiber Length L = 0 km: {key_rate_L0:.2e}")
    print("Optimized Parameters for L = 0 km:")
    for name, value in zip(parameter_names, optimized_params_L0):
        print(f"  {name}: {value:.6f}")
else:
    print("Optimization failed for Fiber Length L = 0 km.")

# %%
with open("training_dataset.json", "r") as f:
    data = json.load(f)

print(data[0])  # Print the first item to inspect its structure

# %%
import json

# Load the dataset
with open("training_dataset.json", "r") as f:
    data = json.load(f)

# Extract the top 100 entries
top_100_entries = data[:100]

# Display the top 100 entries
for idx, entry in enumerate(top_100_entries, 1):
    print(f"Entry {idx}: {entry}")

# %%
import json
import math

# Load the dataset
with open("training_dataset.json", "r") as f:
    data = json.load(f)

# Iterate through each entry
for idx, entry in enumerate(data, 1):
    for key, value in entry.items():
        # Check if value is a number and log10(value) > 0
        if isinstance(value, (int, float)) and value > 1:
            print(f"Entry {idx} Key '{key}': Value = {value}, log10(Value) = {math.log10(value):.2f}")

# %% [markdown]
# ## Current Setup
# 
# 49% of 10,000 iterations completed in 15 minutes. 
# 
# Processing speed: 5.16 iterations per second (it/s). 
# 
# Total Time Estimation (Current Setup): \
# Total iterations: 10,000. \
# Completed iterations:  10,000 \times 0.49 = 4,900 . \
# Time to complete 4,900 iterations: 15 minutes (900 seconds). \
# Estimated total time for 10,000 iterations: 
# 
# $\text{Total time} = \frac{\text{Total iterations}}{\text{Processing speed}} = \frac{10,000}{5.16} \approx 1,937 \text{ seconds (32 minutes)}$
# 
# So, approximately 32 minutes total is needed for the dataset generation with your current setup.
# 
# ## Multiprocessing
# 
# Assumption:
# 12 CPU cores available (based on earlier discussions). \
# Multiprocessing scales linearly with cores (ideal case, no overhead). 
# 
# Parallel Speed Calculation:
# 
# If multiprocessing scales ideally: 
# 
# $\text{Parallel speed} = \text{Single-threaded speed} \times \text{Number of cores}$
# 
# 
# $\text{Parallel speed} = 5.16 \, \text{it/s} \times 12 \approx 61.92 \, \text{it/s}$
# 
# 
# Parallel Time Calculation:
# 
# 
# $\text{Total time (parallel)} = \frac{\text{Total iterations}}{\text{Parallel speed}} = \frac{10,000}{61.92} \approx 161.5 \, \text{seconds (2.7 minutes)}.$
# 
# 

# %% [markdown]
# # Reference
# 1. https://machinelearningmastery.com/dual-annealing-optimization-with-python/
# 2. https://en.wikipedia.org/wiki/Global_optimization
# 3. https://docs.scipy.org/doc/scipy/tutorial/optimize.html


