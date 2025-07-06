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

# %%
import torch

# Check if MPS is available
if torch.backends.mps.is_available():
    device = torch.device("mps")  # Use MPS as the device
    print("Using MPS for GPU acceleration.")
else:
    device = torch.device("cpu")  # Fallback to CPU
    print("MPS is not available. Using CPU.")

# %% [markdown]
# ## Imports

# %%
from math import exp, factorial  # For basic math operations
from scipy.optimize import minimize, dual_annealing, differential_evolution, Bounds  # For optional optimization methods
import matplotlib.pyplot as plt  # For plotting
import numpy as np  # For tensor operations if needed alongside PyTorch
from tqdm import tqdm  # For progress bars
from joblib import Parallel, delayed  # For parallel processing
import os  # For file system operations
import json  # For saving and loading datasets
import time  # For timing operations
import pandas as pd  # For data manipulation
from tabulate import tabulate  # For pretty-printing tables

# PyTorch imports
import torch  # Core PyTorch library

# %% [markdown]
# ## Setup
# 
# Move tensors and models to the GPU using cuda. PyTorch automatically performs operations on the specified device.

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %% [markdown]
# ## Experimental Parameters

# %%
# e_1
# Fiber lengths
Ls = torch.linspace(0.1, 200, 1000)  # Fiber lengths in km
L_BC = Ls
e_1 = L_BC / 100

#e_2
P_dc_value = torch.tensor(6e-7, device=device)   # Dark count probability
Y_0 = P_dc_value
# 2.7*10** -7
# P_dc = 6 * 10 ** (-7)   # given in the paper, discussed with range from 10^-8 to 10^-5
e_2 = -torch.log(Y_0)

# e_3
# Misalignment error probability
# 4*1e-2          # given in the paper, discussed with range from 0 to 0.1
e_mis = torch.tensor(5e-3, device=device)  # Misalignment error probability # given in the paper, discussed with range from 0 to 0.1 
e_d = e_mis
# 0.026 
e_3 = e_d * 100

# e_4
# Detected events
n_X_values = torch.tensor([10 ** s for s in range(6, 11)], dtype=torch.float64, device=device)  # Detected events # Detected events
# n_X_values = torch.tensor([10**s for s in range(6, 11)], dtype=torch.int64)
N = n_X_values
e_4 = torch.log10(N)

# Prepare input combinations
# inputs = [(L, n_X) for L in torch.linspace(0.1, 200, 100) for n_X in n_X_values]

# %% [markdown]
# ## Other Parameters

# %%
alpha = torch.tensor(0.2, device=device)  # Attenuation coefficient (dB/km), given in the paper
eta_Bob = torch.tensor(0.1, device=device)  # Detector efficiency, given in the paper
P_ap = torch.tensor(0.0, device=device)  # After-pulse probability
f_EC = torch.tensor(1.16, device=device)  # Error correction efficiency given in the paper, range around 1.1
epsilon_sec = torch.tensor(1e-10, device=device)  # Security error # is equal to kappa * secrecy length Kl, range around 1e-10 Scalar, as it is a single value throughout the calculations.
epsilon_cor = torch.tensor(1e-15, device=device)  # Correlation error # given in the paper, discussed with range from 0 to 10e-10
# Dark count probability
n_event = torch.tensor(1, device=device)  # For single-photon events
kappa = torch.tensor(1e-15, device=device)  # Security parameter given in the paper

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
    mu_k_values = torch.tensor([mu_1, mu_2, mu_3])
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
from QKD_Functions_Torch import (
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
)

# %%
# Define the `objective` function with `alpha` and other parameters as arguments
def calculate_key_rates_and_metrics(params, L_values, n_X, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event): 
#  mu_k_values, eta_ch_values, p_mu_k_values, p_1 = mu_1, p_2 = mu_2, p_3 = P_mu_1 = 0.65, p_4 = P_mu_2 = 0.3, p_5 =        P_X_value = 5e-3
    mu_1, mu_2, P_mu_1, P_mu_2, P_X_value = params 
    mu_3 = torch.tensor(2e-4, device=L_values.device) # Fixed parameter
    mu_k_values = torch.tensor([mu_1, mu_2, mu_3], device=L_values.device)
    P_mu_3 = 1 - P_mu_1 - P_mu_2
    p_mu_k_values = torch.tensor([P_mu_1, P_mu_2, P_mu_3], device=L_values.device)
       
    P_Z_value = 1 - P_X_value
    # n_X = torch.tensor([10**s for s in range(6, 11)])  # Detected events in X basis: 10^6 to 10^10
    """Objective function to optimize key rate."""
# 1. Channel and system efficiencies
    eta_ch_values = calculate_eta_ch(L_values, alpha)  # Channel transmittance
    eta_sys_values = calculate_eta_sys(eta_Bob, eta_ch_values)  # System transmittance

    # 2. Detection probabilities for each intensity level
    D_mu_k_values = torch.tensor([calculate_D_mu_k(mu_k, eta_sys_values, P_dc_value) for mu_k in mu_k_values],device=L_values.device,)
    # 3. Error rates for each intensity level
    e_mu_k_values = torch.tensor([calculate_e_mu_k(P_dc_value, e_mis, P_ap, D_mu_k, eta_sys_values, mu_k)
                    for D_mu_k, mu_k in zip(D_mu_k_values, mu_k_values)],
        device=L_values.device,)
    # 4. Detection probabilities and events in the X basis
    sum_P_det_mu_X, P_det_mu_1, P_det_mu_2, P_det_mu_3, n_X_total, n_X_mu_1, n_X_mu_2, n_X_mu_3 = calculate_n_X_total(n_event, mu_1, mu_2, mu_3, P_mu_1, P_mu_2, P_mu_3, P_dc_value, eta_sys_values, P_X_value, n_X)
    sqrt_term_n_X = calculate_sqrt_term(n_X, epsilon_sec)  # Uncertainty in X basis

    # Organize detection probabilities and detected events
    n_X_mu_k_values = torch.tensor([n_X_mu_1, n_X_mu_2, n_X_mu_3], device=L_values.device)

    #n_X_total = sum(n_X_mu_k_values)  # Total errors in X basis
    n_X_total = torch.sum(n_X_mu_k_values)

    P_det_mu_values = torch.tensor([P_det_mu_1, P_det_mu_2, P_det_mu_3])

    n_plus_X_mu_1, n_minus_X_mu_1 = calculate_n_pm(mu_1, P_mu_1, n_X_mu_1, sqrt_term_n_X) # m_plus and m_minus for m_X_mu_1

    n_plus_X_mu_2, n_minus_X_mu_2 = calculate_n_pm(mu_2, P_mu_2, n_X_mu_2, sqrt_term_n_X) # m_plus and m_minus for m_X_mu_2

    n_plus_X_mu_3, n_minus_X_mu_3 = calculate_n_pm(mu_3, P_mu_3, n_X_mu_3, sqrt_term_n_X)

    # 5. Total pulses and events in Z basis
    N_values = calculate_N(n_X_total, p_mu_k_values, D_mu_k_values, P_X_value)
    sum_P_det_mu_Z, n_Z_total, n_Z_mu_1, n_Z_mu_2, n_Z_mu_3 = calculate_n_Z_total(N_values, p_mu_k_values, D_mu_k_values, P_Z_value, P_det_mu_values)

    sqrt_term_n_Z = calculate_sqrt_term(n_Z_total, epsilon_sec)  # Uncertainty in Z basis
    
    # Organize detected events in the Z basis
    n_Z_mu_values = torch.tensor([n_Z_mu_1, n_Z_mu_2, n_Z_mu_3], device=L_values.device)
    n_plus_Z_mu_1, n_minus_Z_mu_1 = calculate_n_pm(mu_1, P_mu_1, n_Z_mu_1, sqrt_term_n_Z) # m_plus and m_minus for m_X_mu_1
    n_plus_Z_mu_2, n_minus_Z_mu_2 = calculate_n_pm(mu_2, P_mu_2, n_Z_mu_2, sqrt_term_n_Z) # m_plus and m_minus for m_X_mu_2
    n_plus_Z_mu_3, n_minus_Z_mu_3 = calculate_n_pm(mu_3, P_mu_3, n_Z_mu_3, sqrt_term_n_Z)

    # 7. Security-related terms
    tau_0_values = calculate_tau_n(0, mu_k_values, p_mu_k_values)  # Probability of zero photons
    tau_1_values = calculate_tau_n(1, mu_k_values, p_mu_k_values)  # Probability of one photon

    # 8. Error terms for X basis
    m_X_mu_values = calculate_m_mu_k(e_mu_k_values, p_mu_k_values, N_values, P_X_value)  # List of m_X for intensities
    m_X_mu_1 = m_X_mu_values[0]
    m_X_mu_2 = m_X_mu_values[1]
    m_X_mu_3 = m_X_mu_values[2]
    
    #m_X_mu_values = [m_X_mu_1, m_X_mu_2, m_X_mu_3]
    m_X_mu_values = torch.tensor([m_X_mu_1, m_X_mu_2, m_X_mu_3])

    # m_X_total = sum(m_X_mu_values)  # Total errors in X basis
    m_X_total = torch.sum(m_X_mu_values) 
    
    sqrt_term_m_X = calculate_sqrt_term(m_X_total, epsilon_sec)  # Uncertainty in X error term
    m_plus_X_mu_1, m_minus_X_mu_1 = calculate_m_pm(mu_1, P_mu_1, m_X_mu_1, sqrt_term_m_X) # m_plus and m_minus for m_X_mu_1
    m_plus_X_mu_2, m_minus_X_mu_2 = calculate_m_pm(mu_2, P_mu_2, m_X_mu_2, sqrt_term_m_X) # m_plus and m_minus for m_X_mu_2
    m_plus_X_mu_3, m_minus_X_mu_3 = calculate_m_pm(mu_3, P_mu_3, m_X_mu_3, sqrt_term_m_X)

    # Observed error rate in X basis
    e_obs_X_values = calculate_e_obs(m_X_total, n_X)

    # 9. Error terms for Z basis
    m_Z_mu_values = calculate_m_mu_k(e_mu_k_values, p_mu_k_values, N_values, P_Z_value)
    m_Z_mu_1 = m_Z_mu_values[0]
    m_Z_mu_2 = m_Z_mu_values[1]
    m_Z_mu_3 = m_Z_mu_values[2]

    m_Z_mu_values = torch.tensor([n_X_mu_1, n_X_mu_2, n_X_mu_3])
    m_Z_total = torch.sum(m_Z_mu_values)  # Total errors in Z basi

    sqrt_term_m_Z = calculate_sqrt_term(m_Z_total, epsilon_sec)
    sqrt_term_m_X = calculate_sqrt_term(m_X_total, epsilon_sec)  # Uncertainty in X error term
    m_plus_Z_mu_1, m_minus_Z_mu_1 = calculate_m_pm(mu_1, P_mu_1, m_Z_mu_1, sqrt_term_m_Z)
    m_plus_Z_mu_2, m_minus_Z_mu_2 = calculate_m_pm(mu_2, P_mu_2, m_Z_mu_2, sqrt_term_m_Z)
    m_plus_Z_mu_3, m_minus_Z_mu_3 = calculate_m_pm(mu_3, P_mu_3, m_Z_mu_3, sqrt_term_m_Z)

    # 10. Contributions for single-photon events
    S_X_0_values = calculate_S_0(tau_0_values, mu_2, mu_3, n_minus_X_mu_3, n_plus_X_mu_2)
    S_Z_0_values = calculate_S_0(tau_0_values, mu_2, mu_3, n_minus_Z_mu_3, n_plus_Z_mu_2)

    S_X_1_values = calculate_S_1(tau_1_values, mu_1, mu_2, mu_3,n_minus_X_mu_2, n_plus_X_mu_3, n_plus_X_mu_1, S_X_0_values, tau_0_values)
    S_Z_1_values = calculate_S_1(tau_1_values, mu_1, mu_2, mu_3,n_minus_Z_mu_2, n_plus_Z_mu_3, n_plus_Z_mu_1, S_Z_0_values, tau_0_values)

    # 11. Security bounds and key length
    v_Z_1_values = calculate_v_1(tau_1_values, m_plus_Z_mu_2, m_minus_Z_mu_3, mu_2, mu_3)

    gamma_results = calculate_gamma(epsilon_sec, v_Z_1_values / (S_Z_1_values + 1e-10), S_Z_1_values, S_X_1_values)
    Phi_X_values = calculate_Phi(v_Z_1_values, S_Z_1_values, gamma_results)
    binary_entropy_Phi_values = calculate_h(Phi_X_values)

    # 12. Final key rate and key length
    lambda_EC_values = calculate_lambda_EC(n_X, f_EC, e_obs_X_values)  # Error correction term
    l_calculated_values = calculate_l(S_X_0_values, S_X_1_values, binary_entropy_Phi_values,
                                        lambda_EC_values, epsilon_sec, epsilon_cor)  # Secret key length
    key_rates = calculate_R(l_calculated_values, N_values)  # Secret key rate per pulse

    return (
    key_rates, 
    eta_ch_values, 
    S_X_0_values, 
    S_Z_0_values, 
    S_X_1_values, 
    S_Z_1_values, 
    tau_0_values, 
    tau_1_values, 
    e_mu_k_values,  # Ensure this is included
    e_obs_X_values, 
    v_Z_1_values, 
    gamma_results, 
    Phi_X_values, 
    binary_entropy_Phi_values, 
    lambda_EC_values, 
    l_calculated_values
)

def penalty(key_rates, mu_1, mu_2, mu_3, P_mu_1, P_mu_2, P_mu_3):
    """Penalty function to enforce constraints."""
    # Compute penalties with JAX operations
    # mu_1 > mu_2 + mu_3 # : ensures that mu_1 dominates the sum of mu_2 and mu_3.
    # mu_2 / mu_1 < 1 #: This ensures mu_2 is smaller than mu_1.These conditions are valid, but if one fails, the entire penalty applies, which may over-penalize. Additionally, these are unrelated constraints, so separating them improves clarity.

    penalty_mu1_sum = torch.where(mu_1 > mu_2 + mu_3, 0.0, 1e6)
    penalty_mu2_ratio = torch.where(mu_2 / mu_1 < 1, 0.0, 1e6)
    # This penalty works well for enforcing the sum of probabilities, but the tolerance (1e-10) might be too strict for numerical optimizations, leading to unnecessary penalties.
    penalty_sum = torch.where(torch.abs(P_mu_1 + P_mu_2 + P_mu_3 - 1) < 1e-6, 0.0, 1e6)
    penalty_P_mu_3 = torch.where(P_mu_3 > 0, 0.0, 1e6)
    penalty_mu2_mu3 = torch.where(mu_2 > mu_3, 0.0, 1e6)
    # Sum all penalties
    # Return penalty directly as part of the objective
    total_penalty = penalty_mu1_sum + penalty_mu2_ratio + penalty_sum + penalty_mu2_mu3 + penalty_P_mu_3
    penalized_key_rates = key_rates - total_penalty
    return penalized_key_rates
    
def objective(params, L_values, n_X, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event):
    """
    Objective function with penalty applied to key rates.
    """
    # Unpack parameters
    mu_1, mu_2, P_mu_1, P_mu_2, P_X_value = params
    mu_3 = torch.tensor(2e-4, device=L_values.device)  # Ensure mu_3 is defined
    P_mu_3 = 1 - P_mu_1 - P_mu_2  # Derived value for P_mu_3
    
    # Compute metrics (simulate)
    key_rates, eta_ch_values, S_X_0_values, S_Z_0_values, S_X_1_values, S_Z_1_values, tau_0_values, tau_1_values, e_mu_k_values, e_obs_X_values, v_Z_1_values, gamma_results, Phi_X_values, binary_entropy_Phi_values, lambda_EC_values, l_calculated_values = (
        calculate_key_rates_and_metrics(params, L_values, n_X, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event)
    )
    
    # Apply penalty to key rates
    penalized_key_rates = penalty(key_rates, mu_1, mu_2, mu_3, P_mu_1, P_mu_2, P_mu_3)
    
    # Return all metrics including penalized key rates
    return (
        penalized_key_rates,  # Updated key rates with penalty
        eta_ch_values,
        S_X_0_values,
        S_Z_0_values,
        S_X_1_values,
        S_Z_1_values,
        tau_0_values,
        tau_1_values,
        e_mu_k_values,
        e_obs_X_values,
        v_Z_1_values,
        gamma_results,
        Phi_X_values,
        binary_entropy_Phi_values,
        lambda_EC_values,
        l_calculated_values,
    )

# %%
import torch
# Define bounds for the parameters
bounds = torch.tensor([
    [0.0, 1.0],        # mu_1
    [0.0, 1.0],        # mu_2
    [1e-12, 1.0],      # P_mu_1
    [1e-12, 1.0],      # P_mu_2
    [0.0, 1.0]         # P_X_value
], dtype=torch.float64)

# %%
def sample_within_bounds(bounds):
    """
    Sample a random point within the given bounds.
    """
    return bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * torch.rand(bounds.size(0), dtype=torch.float64)

# %%
def optimize_single_instance(input_params, bounds, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event, device="cpu"):
    """
    Optimize key rates for a given fiber length and n_X value using dual annealing 
    and Nelder-Mead optimization.
    """
    L, n_X = input_params

    # Initialize parameters randomly within bounds
    params = sample_within_bounds(bounds).to(device)
    params.requires_grad = True

    # Define optimizer
    optimizer = torch.optim.Adam([params], lr=0.01)
    
    # Define the loss (negative key rate, since we maximize key rates)
    def loss_fn(params):
        # Negative because we minimize, but key rates need maximization
        penalized_key_rate, *_ = objective(
            params, torch.tensor([L], device=device), torch.tensor([n_X], device=device),
            alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event
        )
        return -penalized_key_rate
    
    # Optimization loop
    max_steps = 500  # Number of optimization steps
    best_params = params.detach().clone()
    best_loss = float("inf")
    for step in range(max_steps):
        optimizer.zero_grad()
        loss = loss_fn(params)
        loss.backward()
        optimizer.step()

        # Clamp parameters within bounds
        with torch.no_grad():
            params.clamp_(bounds[:, 0], bounds[:, 1])

        # Track best result
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_params = params.detach().clone()

    # Final optimized key rate
    optimized_key_rate = -best_loss  # Convert back to key rate (positive)

    return L, n_X, optimized_key_rate, best_params.cpu().numpy()


    # def wrapped_objective(params):
    #     # Negative because we minimize, but key rates need maximization
    #     return -objective(params, L, n_X, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event)[0]

    # # Step 1: Perform global optimization using dual annealing
    # global_result = dual_annealing(func=wrapped_objective, bounds=bounds)

    # # Step 2: Refine results using local optimization (Nelder-Mead)
    # local_result = minimize(
    #     fun=wrapped_objective,
    #     x0=global_result.x,  # Start from global optimization result
    #     method='Nelder-Mead'
    # )

    # # Use the refined results
    # optimized_params = local_result.x
    # optimized_key_rate = -local_result.fun  # Convert back to key rate (positive)

    # return L, n_X, optimized_key_rate, optimized_params

# %%
from joblib import Parallel, delayed

# Parallelize on CPU while GPU handles computations
def optimize_single_instance_on_gpu(input_params, bounds, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event):
    L, n_X = input_params
    # Convert to GPU
    L = torch.tensor(L, device=device, dtype=torch.float64)
    n_X = torch.tensor(n_X, device=device, dtype=torch.float64)
    # Perform optimization on GPU (dummy example)
    return L.item(), n_X.item(), (L + n_X).item(), [L.item(), n_X.item(), 0.1, 0.1, 0.5]

# Generate dataset with parallel CPU processing
inputs = [(L.item(), n_X.item()) for L in Ls for n_X in n_X_values]
results = Parallel(n_jobs=12)(
    delayed(optimize_single_instance_on_gpu)(
        input_params, bounds, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event
    )
    for input_params in inputs
)

# %%
# Define input parameters for a single instance
single_input = (Ls[0].item(), n_X_values[0].item())  # Example: first fiber length and first n_X value

# Measure start time
start_time = time.time()

# Run the optimization for the single instance
L, n_X, optimized_key_rate, optimized_params = optimize_single_instance(
    single_input, bounds, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event, device="cuda" if torch.cuda.is_available() else "cpu"
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
def generate_dataset(Ls, n_X_values, bounds, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event, device=None):
    """
    Generate a dataset by optimizing key rates for various fiber lengths and n_X values.
    """
    # Create input combinations
    inputs = [(L.item(), n_X.item()) for L in Ls for n_X in n_X_values]

    print("Generating dataset...")
    results = []

    def process_batch(batch):
        batch_results = []
        for input_params in batch:
            L, n_X, penalized_key_rate, optimized_params = optimize_single_instance_on_gpu(
                input_params, bounds, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event
            )
            batch_results.append((L, n_X, penalized_key_rate, optimized_params))
        return batch_results

    # Split inputs into batches
    batch_size = 100
    batches = [inputs[i:i+batch_size] for i in range(0, len(inputs), batch_size)]


    # Progress bar for monitoring
    with tqdm(total=len(batches), desc="Generating Dataset") as progress_bar:
        results = Parallel(n_jobs=12)(
            delayed(process_batch)(batch) for batch in batches
        )
        progress_bar.update(len(batches))

    # Flatten results
    results = [item for sublist in results for item in sublist]

    # Process results into a dataset
    dataset = []
    for L, n_X, penalized_key_rate, optimized_params in results:
        # Extract optimized parameters (ensure the order matches the bounds setup)
        mu_1, mu_2, P_mu_1, P_mu_2, P_X_value = optimized_params

        # Compute normalized parameters
        e_1 = L / 100  # Normalize fiber length
        e_2 = -torch.log10(P_dc_value).item()  # Normalize dark count probability
        e_3 = (e_mis * 100).item()  # Normalize misalignment error probability
        e_4 = torch.log10(torch.tensor(n_X, dtype=torch.float64, device=device)).item()  # Normalize number of pulses

        # Append processed data
        dataset.append({
            "e_1": e_1,
            "e_2": e_2,
            "e_3": e_3,
            "e_4": e_4,
            "key_rate": penalized_key_rate,
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
# Experiment-specific parameters

# Define parameters as PyTorch tensors
P_dc_value = torch.tensor(6e-7, device=device, dtype=torch.float64)  # Dark count probability
e_mis = torch.tensor(5e-3, device=device, dtype=torch.float64)       # Misalignment error probability
alpha = torch.tensor(0.2, device=device, dtype=torch.float64)        # Attenuation coefficient (dB/km), given in the paper
eta_Bob = torch.tensor(0.1, device=device, dtype=torch.float64)      # Detector efficiency, given in the paper
P_ap = torch.tensor(1e-6, device=device, dtype=torch.float64)        # After-pulse probability # 4*1e-2          # given in the paper, discussed with range from 0 to 0.1
f_EC = torch.tensor(1.16, device=device, dtype=torch.float64)        # Error correction efficiency # given in the paper, range around 1.1
epsilon_sec = torch.tensor(1e-10, device=device, dtype=torch.float64) # Security error # is equal to kappa * secrecy length Kl, range around 1e-10 Scalar, as it is a single value throughout the calculations.
epsilon_cor = torch.tensor(1e-15, device=device, dtype=torch.float64) # Correlation error
n_event = torch.tensor(1, device=device, dtype=torch.float64)         # For single photon event
kappa = torch.tensor(1e-15, device=device, dtype=torch.float64)       # given in the papere_1 # given in the paper, discussed with range from 0 to 10e-10

# Define parameter space
Ls = torch.linspace(0.1, 220, 100, device=device, dtype=torch.float64)  # Fiber lengths
n_X_values = torch.logspace(6, 10, 8, device=device, dtype=torch.float64)  # Detected events


# 
# torch.linspace(0.1, 220, 10):
# 	•	Generates 10 equally spaced values between 0.1 and 220. For example: [0.1, 27.88, 55.66, ..., 220].
# 	•	Represents 10 fiber lengths to evaluate, rather than 1000.
# np.logspace(6, 8, 5):
# 	•	Generates 5 logarithmically spaced values between 10^6 and 10^8. For example: [1e6, 3.16e6, 1e7, ..., 1e8].
# 	•	Represents the number of detected events (n_X) in a smaller range with fewer steps.

import cProfile
cProfile.run("""
optimize_single_instance(
    (Ls[0].item(), n_X_values[0].item()), bounds, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event, device=device
)
""")


# Measure total dataset generation time
start_time = time.time()

dataset = generate_dataset(
    Ls, n_X_values, bounds, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event, device=device
)

end_time = time.time()
print(f"Dataset generation completed in {(end_time - start_time) / 60:.2f} minutes.")

# Save to JSON
output_filename = "training_dataset.json"
with open(output_filename, "w") as f:
    json.dump(dataset, f)
print(f"Dataset saved as '{output_filename}'.")

# %%
import torch
import matplotlib.pyplot as plt
import json

# Load the dataset
with open("training_dataset.json", "r") as f:
    data = json.load(f)

# Extract fiber lengths and key rates
e_1 = torch.tensor([item["e_1"] * 100 for item in data], dtype=torch.float64)  # Denormalize fiber lengths (convert to km)
key_rate = torch.tensor([item["key_rate"] for item in data], dtype=torch.float64)  # Correct key name

# Extract optimized parameters
mu_1 = torch.tensor([item["optimized_params"]["mu_1"] for item in data], dtype=torch.float64)  # Access nested keys
mu_2 = torch.tensor([item["optimized_params"]["mu_2"] for item in data], dtype=torch.float64)
P_mu_1 = torch.tensor([item["optimized_params"]["P_mu_1"] for item in data], dtype=torch.float64)
P_mu_2 = torch.tensor([item["optimized_params"]["P_mu_2"] for item in data], dtype=torch.float64)
P_X_value = torch.tensor([item["optimized_params"]["P_X_value"] for item in data], dtype=torch.float64)

# Sort by fiber length for smooth plotting
sorted_indices = torch.argsort(e_1)
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
plt.plot(e_1_sorted.cpu(), torch.log10(torch.clamp(key_rate_sorted, min=1e-10)).cpu(), label="Penalized Key Rate (log10)")
plt.xlabel("Fiber Length (km)")
plt.ylabel("log10(Penalized Key Rate)")
plt.title("Penalized Key Rate vs Fiber Length")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend()

# Right plot: Optimized Parameters
plt.subplot(1, 2, 2)
plt.plot(e_1_sorted.cpu(), mu_1_sorted.cpu(), label="mu_1")
plt.plot(e_1_sorted.cpu(), mu_2_sorted.cpu(), label="mu_2")
plt.plot(e_1_sorted.cpu(), P_mu_1_sorted.cpu(), label="P_mu_1")
plt.plot(e_1_sorted.cpu(), P_mu_2_sorted.cpu(), label="P_mu_2")
plt.plot(e_1_sorted.cpu(), P_X_value_sorted.cpu(), label="P_X_value")
plt.xlabel("Fiber Length (km)")
plt.ylabel("Optimized Parameters")
plt.title("Optimized Parameters vs Fiber Length")
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend()

plt.tight_layout()
plt.show()

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
print("\nTop 100 entries:")
for idx, entry in enumerate(top_100_entries, 1):
    print(f"Entry {idx}: {entry}")

# Optional: Flatten the JSON structure and save as a CSV file
df = pd.json_normalize(data, sep='_')  # Flatten the JSON structure

# Convert numeric data to PyTorch tensors where applicable
for col in df.select_dtypes(include=['float', 'int']).columns:
    df[col] = df[col].apply(lambda x: torch.tensor(x, dtype=torch.float64))

# Save the DataFrame as a CSV file
output_csv_file = "training_dataset.csv"
df.to_csv(output_csv_file, index=False)

print(f"\nCSV file saved as '{output_csv_file}'.")

# import json
# import pandas as pd

# # Load the JSON data
# with open("training_dataset.json", "r") as f:
#     data = json.load(f)

# # Flatten the JSON structure
# df = pd.json_normalize(data, sep='_')

# # Save the DataFrame as a CSV file
# output_csv_file = "training_dataset.csv"
# df.to_csv(output_csv_file, index=False)

# print(f"CSV file saved as {output_csv_file}.")

# %%
# bounds = Bounds([1e-6] * 5, [1.0] * 5)  # Example bounds
# input_params = (100, 1e8)  # Example fiber length and n_X
# result = optimize_single_instance(
#     input_params, bounds, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event
# )
# print(result)

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


