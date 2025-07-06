# %%
# PyTorch and tensor operations
import torch

# Mathematical operations
from math import exp, factorial

# Plotting
import matplotlib.pyplot as plt

# Progress tracking
from tqdm import tqdm

# File handling and JSON processing
import json
import os
import time

# Data manipulation
import pandas as pd

# Table formatting for better visualization
from tabulate import tabulate

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
# ## Optimal Parameters

# %%
# p, optimal parameters
p_1 = mu_1 = 6e-1
p_2 = mu_2 = 2e-1
mu_3 = 2e-4
mu_k_values = [mu_1, mu_2, mu_3]
p_3 = P_mu_1 = 0.65
p_4 = P_mu_2 = 0.3
P_mu_3 = 1 - P_mu_1 - P_mu_2
p_mu_k_values = [P_mu_1, P_mu_2, P_mu_3]
p_5 = P_X_value = 5e-3
P_Z_value = 1 - P_X_value
# def optimal_parameters(params):
#     mu_1, mu_2, P_mu_1, P_mu_2, P_X_value = params
#     mu_3 = 2e-4
#     P_mu_3 = 1 - P_mu_1 - P_mu_2
#     P_Z_value = 1 - P_X_value
#     mu_k_values = torch.tensor([mu_1, mu_2, mu_3])
#     return params, mu_3, P_mu_3, P_Z_value, mu_k_values

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
import torch

# Check for MPS (Metal Performance Shaders) support
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS for GPU acceleration.")
else:
    device = torch.device("cpu")
    print("MPS is not available. Using CPU.")

# Define metric information
metric_info = {
    "eta_ch": ("Channel Transmittance $\eta_{ch}$", "Channel Transmittance vs Fiber Length"),
    "S_X_0": ("$S_{X_0}$", "Single-photon Events $S_{X_0}$ vs Fiber Length"),
    "S_Z_0": ("$S_{Z_0}$", "Single-photon Events $S_{Z_0}$ vs Fiber Length"),
    "tau_0": ("Tau 0", "Tau 0 vs Fiber Length"),
    "e_obs_X": ("$e_{obs,X}$", "Observed Error Rate $e_{obs,X}$ vs Fiber Length"),
    "S_X_1": ("$S_{X_1}$", "Single-photon Events $S_{X_1}$ vs Fiber Length"),
    "S_Z_1": ("$S_{Z_1}$", "Single-photon Events $S_{Z_1}$ vs Fiber Length"),
    "tau_1": ("Tau 1", "Tau 1 vs Fiber Length"),
    "v_Z_1": ("$v_{Z_1}$", "$v_{Z_1}$ vs Fiber Length"),
    "gamma": ("Gamma", "Gamma vs Fiber Length"),
    "Phi_X": ("$\Phi_{X}$", "$\Phi_{X}$ vs Fiber Length"),
    "binary_entropy_Phi": ("Binary Entropy of $\Phi$", "Binary Entropy of $\Phi$ vs Fiber Length"),
    "lambda_EC": ("Lambda EC", "Lambda EC vs Fiber Length"),
    "l_calculated": ("Calculated Secret Key Length $l$", "Calculated Secret Key Length $l$ vs Fiber Length"),
    "key_rates": ("Secret Key Rate per Pulse (bit)", "Secret Key Rate vs Fiber Length"),
}

# Define range of `n_X` values and fiber lengths
Ls = torch.linspace(0.1, 220, 1000, device=device, dtype=torch.float64)  # Fiber lengths
params = (mu_1, mu_2, P_mu_1, P_mu_2, P_X_value)  # Predefined parameters

# Result storage dictionary for different `n_X` values
results = {n_X.item(): {metric: [] for metric in metric_info.keys()} for n_X in n_X_values.to(device)}
for n_X in n_X_values.to(device):
    results[n_X.item()]["fiber_lengths"] = []  # Explicitly include fiber_lengths

# Main calculation loop for each `n_X` value
for n_X in n_X_values.to(device):
    for L_values in Ls:
        result = objective(
            params, 
            L_values, 
            n_X, 
            alpha, 
            eta_Bob, 
            P_dc_value, 
            epsilon_sec, 
            epsilon_cor, 
            f_EC, 
            e_mis, 
            P_ap, 
            n_event,
        )
        (
            penalized_key_rates,
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
        ) = result

        # Append metrics to the results for the current n_X
        n_X_key = n_X.item()
        results[n_X_key]["fiber_lengths"].append(L_values.item())
        results[n_X_key]["key_rates"].append(penalized_key_rates.item())
        results[n_X_key]["eta_ch"].append(eta_ch_values.item())
        results[n_X_key]["S_X_0"].append(S_X_0_values.item())
        results[n_X_key]["S_Z_0"].append(S_Z_0_values.item())
        results[n_X_key]["S_X_1"].append(S_X_1_values.item())
        results[n_X_key]["S_Z_1"].append(S_Z_1_values.item())
        results[n_X_key]["tau_0"].append(tau_0_values.item())
        results[n_X_key]["tau_1"].append(tau_1_values.item())
        results[n_X_key]["e_obs_X"].append(e_obs_X_values.item())
        results[n_X_key]["v_Z_1"].append(v_Z_1_values.item())
        results[n_X_key]["gamma"].append(gamma_results.item())
        results[n_X_key]["Phi_X"].append(Phi_X_values.item())
        results[n_X_key]["binary_entropy_Phi"].append(binary_entropy_Phi_values.item())
        results[n_X_key]["lambda_EC"].append(lambda_EC_values.item())
        results[n_X_key]["l_calculated"].append(l_calculated_values.item())

# metric_info = {
#     "eta_ch": ("Channel Transmittance $\eta_{ch}$", "Channel Transmittance vs Fiber Length"),
#     "S_X_0": ("$S_{X_0}$", "Single-photon Events $S_{X_0}$ vs Fiber Length"),
#     "S_Z_0": ("$S_{Z_0}$", "Single-photon Events $S_{Z_0}$ vs Fiber Length"),
#     "tau_0": ("Tau 0", "Tau 0 vs Fiber Length"),
#     "e_obs_X": ("$e_{obs,X}$", "Observed Error Rate $e_{obs,X}$ vs Fiber Length"),
#     "S_X_1": ("$S_{X_1}$", "Single-photon Events $S_{X_1}$ vs Fiber Length"),
#     "S_Z_1": ("$S_{Z_1}$", "Single-photon Events $S_{Z_1}$ vs Fiber Length"),
#     "tau_1": ("Tau 1", "Tau 1 vs Fiber Length"),
#     "v_Z_1": ("$v_{Z_1}$", "$v_{Z_1}$ vs Fiber Length"),
#     "gamma": ("Gamma", "Gamma vs Fiber Length"),
#     "Phi_X": ("$\Phi_{X}$", "$\Phi_{X}$ vs Fiber Length"),
#     "binary_entropy_Phi": ("Binary Entropy of $\Phi$", "Binary Entropy of $\Phi$ vs Fiber Length"),
#     "lambda_EC": ("Lambda EC", "Lambda EC vs Fiber Length"),
#     "l_calculated": ("Calculated Secret Key Length $l$", "Calculated Secret Key Length $l$ vs Fiber Length"),
#     "key_rates": ("Secret Key Rate per Pulse (bit)", "Secret Key Rate vs Fiber Length")
# }

# # Define range of `n_X` values and fiber lengths

# Ls = jnp.linspace(0.1, 220, 1000)  # Fiber lengths from 0.1 km to 300 km, 100 points
# params = (mu_1, mu_2, P_mu_1, P_mu_2, P_X_value)

# # # Result storage dictionary for different `n_X` values
# # Result storage dictionary for different `n_X` values
# results = {n_X: {} for n_X in n_X_values}

# # Main calculation loop for each `n_X` value
# for n_X in n_X_values:
#     results[n_X] = {metric: [] for metric in metric_info.keys()}
#     results[n_X]["fiber_lengths"] = []  # Explicitly include fiber_lengths

#     # Loop over each fiber length
#     for L_values in Ls:
#          # (key_rates, eta_ch_values, S_X_0_values, S_Z_0_values, S_X_1_values, S_Z_1_values, tau_0_values, tau_1_values, e_mu_k_values, e_obs_X_values,  v_Z_1_values, gamma_results, Phi_X_values, binary_entropy_Phi_values, lambda_EC_values, l_calculated_values)
#         # result = objective(params, L_values, n_X, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event)
#         # (
#         # key_rates, eta_ch_values, S_X_0_values, S_Z_0_values, S_X_1_values, S_Z_1_values,
#         # tau_0_values, tau_1_values, e_mu_k_values, e_obs_X_values, v_Z_1_values,
#         # gamma_results, Phi_X_values, binary_entropy_Phi_values, lambda_EC_values,
#         # l_calculated_values
#         # ) = result
#         result = objective(params, L_values, n_X, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event)
#         (
#         penalized_key_rates,  # Updated key rates with penalty
#         eta_ch_values,
#         S_X_0_values,
#         S_Z_0_values,
#         S_X_1_values,
#         S_Z_1_values,
#         tau_0_values,
#         tau_1_values,
#         e_mu_k_values,
#         e_obs_X_values,
#         v_Z_1_values,
#         gamma_results,
#         Phi_X_values,
#         binary_entropy_Phi_values,
#         lambda_EC_values,
#         l_calculated_values,
#     ) = result

#         # Append metrics to the results for the current n_X
#         results[n_X]["fiber_lengths"].append(L_values)
#         results[n_X]["key_rates"].append(penalized_key_rates)
#         results[n_X]["eta_ch"].append(eta_ch_values)
#         results[n_X]["S_X_0"].append(S_X_0_values)
#         results[n_X]["S_Z_0"].append(S_Z_0_values)
#         results[n_X]["S_X_1"].append(S_X_1_values)
#         results[n_X]["S_Z_1"].append(S_Z_1_values)
#         results[n_X]["tau_0"].append(tau_0_values)
#         results[n_X]["tau_1"].append(tau_1_values)
#         results[n_X]["e_obs_X"].append(e_obs_X_values)
#         results[n_X]["v_Z_1"].append(v_Z_1_values)
#         results[n_X]["gamma"].append(gamma_results)
#         results[n_X]["Phi_X"].append(Phi_X_values)
#         results[n_X]["binary_entropy_Phi"].append(binary_entropy_Phi_values)
#         results[n_X]["lambda_EC"].append(lambda_EC_values)
#         results[n_X]["l_calculated"].append(l_calculated_values)


# %%
for n_X, data in results.items():
    print(f"n_X: {n_X}")
    for key, value in data.items():
        print(f"  {key}: {type(value)}, Length: {len(value)}")

# %%
import torch
import os
import matplotlib.pyplot as plt
import numpy as np

# Ensure `output_dir` is defined
output_dir = 'analytical_result'
os.makedirs(output_dir, exist_ok=True)

# Plotting loop
for metric, (ylabel, title) in metric_info.items():
    plt.figure(figsize=(12, 8))
    for n_X, data in results.items():
        if metric in data and "fiber_lengths" in data:  # Check both keys exist
            fiber_lengths = torch.tensor(data["fiber_lengths"], dtype=torch.float64)
            metric_values = torch.tensor(data[metric], dtype=torch.float64)
            plt.plot(fiber_lengths.cpu().numpy(), metric_values.cpu().numpy(), label=f'$n_X = {n_X:.0e}$')
    plt.xlabel('Fiber Length (km)')
    plt.ylabel(ylabel)
    plt.yscale('log')
    plt.title(title)
    plt.legend()
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)

    # Save the plot
    filename = os.path.join(output_dir, f'{metric}.png')
    plt.savefig(filename)
    plt.show()

# Define the number of rows and columns for the grid of subplots
num_metrics = len(metric_info)
rows = int(np.ceil(np.sqrt(num_metrics)))  # Rows of subplots
cols = int(np.ceil(num_metrics / rows))   # Columns of subplots

# Ensure the directory exists
output_dir = 'analytical_result'
os.makedirs(output_dir, exist_ok=True)

# Close all previous figures to avoid interference
plt.close('all')

fig, axes = plt.subplots(rows, cols, figsize=(24, 18), sharex=True)

# Flatten the axes array for easier indexing
axes = axes.flatten()

# Plot each metric in its respective subplot
for idx, (metric, (ylabel, title)) in enumerate(metric_info.items()):
    ax = axes[idx]
    for n_X, data in results.items():
        if metric in data:
            fiber_lengths = torch.tensor(data["fiber_lengths"], dtype=torch.float64)
            metric_values = torch.tensor(data[metric], dtype=torch.float64)
            ax.plot(fiber_lengths.cpu().numpy(), metric_values.cpu().numpy(), label=f'$n_X = {n_X:.0e}$')
    ax.set_xlabel('Fiber Length (km)')
    ax.set_ylabel(ylabel)
    ax.set_yscale('log')  # Log scale for y-axis
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", linestyle='--', linewidth=0.5)

# Hide any unused subplots
for idx in range(len(metric_info), len(axes)):
    fig.delaxes(axes[idx])

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the grid of plots before showing it
filename = os.path.join(output_dir, 'all_metrics_grid.png')
plt.savefig(filename)

# Display the plot
plt.show()

# # Ensure `output_dir` is defined
# import os
# output_dir = 'analytical_result'
# os.makedirs(output_dir, exist_ok=True)

# # Plotting loop
# for metric, (ylabel, title) in metric_info.items():
#     plt.figure(figsize=(12, 8))
#     for n_X, data in results.items():
#         if metric in data and "fiber_lengths" in data:  # Check both keys exist
#             plt.plot(data["fiber_lengths"], data[metric], label=f'$n_X = {n_X:.0e}$')
#     plt.xlabel('Fiber Length (km)')
#     plt.ylabel(ylabel)
#     plt.yscale('log')
#     plt.title(title)
#     plt.legend()
#     plt.grid(True, which="both", linestyle='--', linewidth=0.5)

#     # Save the plot
#     filename = os.path.join(output_dir, f'{metric}.png')
#     plt.savefig(filename)
#     plt.show()

# # Define the number of rows and columns for the grid of subplots
# num_metrics = len(metric_info)
# rows = int(np.ceil(np.sqrt(num_metrics)))  # Rows of subplots
# cols = int(np.ceil(num_metrics / rows))   # Columns of subplots

# # Ensure the directory exists
# output_dir = 'analytical_result'
# os.makedirs(output_dir, exist_ok=True)

# # Close all previous figures to avoid interference
# plt.close('all')

# fig, axes = plt.subplots(rows, cols, figsize=(24, 18), sharex=True)

# # Flatten the axes array for easier indexing
# axes = axes.flatten()

# # Plot each metric in its respective subplot
# for idx, (metric, (ylabel, title)) in enumerate(metric_info.items()):
#     ax = axes[idx]
#     for n_X, data in results.items():
#         if metric in data:
#             ax.plot(data["fiber_lengths"], data[metric], label=f'$n_X = {n_X:.0e}$')
#     ax.set_xlabel('Fiber Length (km)')
#     ax.set_ylabel(ylabel)
#     ax.set_yscale('log')  # Log scale for y-axis
#     ax.set_title(title)
#     ax.legend()
#     ax.grid(True, which="both", linestyle='--', linewidth=0.5)

# # Hide any unused subplots
# for idx in range(len(metric_info), len(axes)):
#     fig.delaxes(axes[idx])

# # Adjust layout to prevent overlap
# plt.tight_layout()

# # Save the grid of plots before showing it
# filename = os.path.join(output_dir, 'all_metrics_grid.png')
# plt.savefig(filename)

# # Display the plot
# plt.show()

# %%


# %%
# output_dir = 'analytical_result'
# os.makedirs(output_dir, exist_ok=True)

# # Plotting loop
# for metric, (ylabel, title) in metric_info.items():
#     plt.figure(figsize=(12, 8))
#     for n_X, data in results.items():
#         if metric in data:
#             plt.plot(data["fiber_lengths"], data[metric], label=f'$n_X = {n_X:.0e}$')
#     plt.xlabel('Fiber Length (km)')
#     plt.ylabel(ylabel)
#     plt.yscale('log')
#     plt.title(title)
#     plt.legend()
#     plt.grid(True, which="both", linestyle='--', linewidth=0.5)

#     # Save the plot
#     filename = os.path.join(output_dir, f'{metric}.png')
#     plt.savefig(filename)
#     plt.show()

#     # Define the number of rows and columns for the grid of subplots
# num_metrics = len(metric_info)
# rows = int(np.ceil(np.sqrt(num_metrics)))  # Rows of subplots
# cols = int(np.ceil(num_metrics / rows))   # Columns of subplots

# # Ensure the directory exists
# output_dir = 'analytical_result'
# os.makedirs(output_dir, exist_ok=True)

# # Close all previous figures to avoid interference
# plt.close('all')

# fig, axes = plt.subplots(rows, cols, figsize=(24, 18), sharex=True)

# # Flatten the axes array for easier indexing
# axes = axes.flatten()

# # Plot each metric in its respective subplot
# for idx, (metric, (ylabel, title)) in enumerate(metric_info.items()):
#     ax = axes[idx]
#     for n_X, data in results.items():
#         if metric in data:
#             ax.plot(data["fiber_lengths"], data[metric], label=f'$n_X = {n_X:.0e}$')
#     ax.set_xlabel('Fiber Length (km)')
#     ax.set_ylabel(ylabel)
#     ax.set_yscale('log')  # Log scale for y-axis
#     ax.set_title(title)
#     ax.legend()
#     ax.grid(True, which="both", linestyle='--', linewidth=0.5)                          

# # Hide any unused subplots
# for idx in range(num_metrics, len(axes)):
#     fig.delaxes(axes[idx])

# # Adjust layout to prevent overlap
# plt.tight_layout()

# # Save the grid of plots before showing it
# filename = os.path.join(output_dir, 'all_metrics_grid.png')
# plt.savefig(filename)

# # Display the plot
# plt.show()


# %%
# # Dictionary to define y-axis labels and titles for each metric
# metric_info = {
#     "eta_ch": ("Channel Transmittance $\eta_{ch}$", "Channel Transmittance vs Fiber Length"),
#     "S_X_0": ("$S_{X_0}$", "Single-photon Events $S_{X_0}$ vs Fiber Length"),
#     "S_Z_0": ("$S_{Z_0}$", "Single-photon Events $S_{Z_0}$ vs Fiber Length"),
#     "tau_0": ("Tau 0", "Tau 0 vs Fiber Length"),
#     "e_obs_X": ("$e_{obs,X}$", "Observed Error Rate $e_{obs,X}$ vs Fiber Length"),
#     "S_X_1": ("$S_{X_1}$", "Single-photon Events $S_{X_1}$ vs Fiber Length"),
#     "S_Z_1": ("$S_{Z_1}$", "Single-photon Events $S_{Z_1}$ vs Fiber Length"),
#     "tau_1": ("Tau 1", "Tau 1 vs Fiber Length"),
#     "v_Z_1": ("$v_{Z_1}$", "$v_{Z_1}$ vs Fiber Length"),
#     "gamma": ("Gamma", "Gamma vs Fiber Length"),
#     "Phi_X": ("$\Phi_{X}$", "$\Phi_{X}$ vs Fiber Length"),
#     "binary_entropy_Phi": ("Binary Entropy of $\Phi$", "Binary Entropy of $\Phi$ vs Fiber Length"),
#     "lambda_EC": ("Lambda EC", "Lambda EC vs Fiber Length"),
#     "l_calculated": ("Calculated Secret Key Length $l$", "Calculated Secret Key Length $l$ vs Fiber Length"),
#     "key_rates": ("Secret Key Rate per Pulse (bit)", "Secret Key Rate vs Fiber Length")
# }

# %%
# def validate_parameters_and_conditions():
#     """
#     Validates key conditions and parameter ranges for decoy-state QKD setup.

#     Returns:
#     - bool: True if all conditions and range checks are satisfied, else False.
#     """
#     # Core conditions for parameter relationships
#     conditions = {
#         "Condition 1: mu_1 > mu_2 + mu_3": mu_1 > mu_2 + mu_3,
#         "Condition 2: mu_2 > mu_3 >= 0": mu_2 > mu_3 >= 0,
#         "Condition 3: P_mu values sum to 1": abs(P_mu_1 + P_mu_2 + P_mu_3 - 1) < 1e-5,
#         "Condition 4: P_X + P_Z sum to 1": abs(P_X_value + P_Z_value - 1) < 1e-5
#     }
    
#     # Print and evaluate each condition
#     all_conditions_passed = True
#     for condition, is_satisfied in conditions.items():
#         status = "Pass" if is_satisfied else "Fail"
#         print(f"{condition}: {status}")
#         all_conditions_passed &= is_satisfied  # Update the final status

#     # Bound checks for parameters
#     bounds = {
#         "Minimum Fiber Length": (min(Ls), (0.1, 200)),
#         "Maximum Fiber Length": (max(Ls), (0.1, 200)),
#         "Attenuation Coefficient (alpha)": (alpha, (0.1, 1)),
#         "Dark Count Probability (P_dc)": (P_dc_value, (1e-8, 1e-5)),
#         "After pulse Probability (P_ap)": (P_ap, (0, 0.1)),
#         "System Transmittance (eta_sys)": (eta_sys_values, (1e-6, 1)),
#         "Channel Transmittance (eta_ch)": (eta_ch_values, (1e-6, 1)),
#         "Detector Efficiency (eta_Bob)": (eta_Bob, (0, 1)),
#         "Secrecy Parameter (epsilon_sec)": (epsilon_sec, (1e-10, 1)),
#         "Correctness Parameter (epsilon_cor)": (epsilon_cor, (1e-15, 1)),
#         "Error Correction Efficiency (f_EC)": (f_EC, (1, 2)),
#         "Detected Events in X Basis (n_X)": (n_X_values, (1e9, 1e11)),
#         "Detected Events in Z Basis (n_Z)": (n_Z_mu_values, (1e8, 1e11)),  # List validation
#         "Total Events in Z Basis (n_Z_total)": (n_Z_total, (1e9, 1e11)),  # Scalar validation
#     }

#     # Check each parameter's bounds and store out-of-bound values
#     all_bounds_passed = True
#     out_of_bound_params = []
#     for name, (value, range_) in bounds.items():
#         if isinstance(value, (list, np.ndarray, jnp.ndarray)):  # Handle lists/arrays
#             out_of_bounds = [
#                 (i, v) for i, v in enumerate(value) if not (range_[0] <= v <= range_[1])
#             ]
#             if out_of_bounds:
#                 print(f"{name} out of bounds:")
#                 for idx, v in out_of_bounds:
#                     print(f"  - Element {idx}: {v} (Expected range: {range_})")
#                 out_of_bound_params.append((name, out_of_bounds))
#                 all_bounds_passed = False
#             else:
#                 print(f"{name}: All elements within bounds.")
#         elif isinstance(value, (jnp.ndarray, np.ndarray)):  # Scalar-like array
#             within_bounds = jnp.logical_and(range_[0] <= value, value <= range_[1]).all()
#             status = "within bounds" if within_bounds else "out of bounds"
#             print(f"{name}: {value} ({status}) - Expected range: {range_}")
#             if not within_bounds:
#                 out_of_bound_params.append((name, value, range_))
#                 all_bounds_passed = False
#         else:  # Handle single scalar values
#             within_bounds = range_[0] <= value <= range_[1]
#             status = "within bounds" if within_bounds else "out of bounds"
#             print(f"{name}: {value} ({status}) - Expected range: {range_}")
#             if not within_bounds:
#                 out_of_bound_params.append((name, value, range_))
#                 all_bounds_passed = False

#     # Print out-of-bound values, if any
#     if out_of_bound_params:
#         print("\nThe following parameters are out of bounds:")
#         for param in out_of_bound_params:
#             if isinstance(param[1], list):  # List-like parameter
#                 name, out_of_bounds = param
#                 for idx, value in out_of_bounds:
#                     print(f"  - {name} (Element {idx}): {value} (Expected range: {bounds[name][1]})")
#             else:  # Scalar parameter
#                 name, value, range_ = param
#                 print(f"  - {name}: {value} (Expected range: {range_})")

#     # Final validation status
#     if all_conditions_passed and all_bounds_passed:
#         print("\n✅ All conditions and parameter ranges are within expected bounds.")
#         return True
#     else:
#         print("\n❌ One or more conditions/parameters are out of bounds.")
#         return False

# %%
# # Parameter values
# print(f"Fiber length range (L): {min(Ls)} km to {max(Ls)} km")
# print(f"Attenuation coefficient (alpha): {alpha}")
# print(f"Dark count probability (P_dc): {P_dc_value}")
# print(f"Misalignment error probability (e_mis): {e_mis}")
# print(f"After pulse probability (P_ap): {P_ap}")
# print(f"System transmittance (eta_sys_values): {eta_sys_values}")
# print(f"Channel transmittance (eta_ch_values): {eta_ch_values}")
# print(f"Detector efficiency (eta_Bob): {eta_Bob}")
# print(f"Secrecy parameter (epsilon_sec): {epsilon_sec}")
# print(f"Correctness parameter (epsilon_cor): {epsilon_cor}")
# print(f"Error correction efficiency (f_EC): {f_EC}")
# print(f"Detected events in X basis (n_X): {n_X}")
# print(f"Detected events in Z basis (n_Z_mu_values): {n_Z_mu_values}")
# print(f"Probability of X basis (P_X_value): {P_X_value}")
# print(f"Probability of Z basis (P_Z_value): {P_Z_value}")
# print(f"Sum of intensities probabilities (P_mu_1 + P_mu_2 + P_mu_3): {P_mu_1 + P_mu_2 + P_mu_3}")
# print(f"Total probability of detection in X basis (sum_P_det_mu_X): {sum_P_det_mu_X}")
# print(f"Total probability of detection in Z basis (sum_P_det_mu_Z): {sum_P_det_mu_Z}")
# print(f"Total events in X basis (n_X_values): {n_X_mu_k_values}")
# print(f"Total events in Z basis (n_X_total): {n_X_total}\n")

# print(f"Total errors in X basis (m_X_values): {m_X_mu_values}")
# print(f"Total errors in X basis (m_X_total): {m_X_total}")
# print(f"Total errors in Z basis (m_Z_values): {m_Z_mu_values}")
# print(f"Total errors in Z basis (m_Z_total): {m_Z_total}\n")

# # Tau and Square Root Terms
# print(f"Poisson probability for zero photons (tau_0_values): {tau_0_values}")
# print(f"Poisson probability for one photon (tau_1_values): {tau_1_values}")
# print(f"Square root term of n in X basis (sqrt_term_n_X): {sqrt_term_n_X}")
# print(f"Square root term of n in Z basis (sqrt_term_n_Z): {sqrt_term_n_Z}\n")
# print(f"Square root term of m in X basis (sqrt_term_m_X): {sqrt_term_m_X} \n")
# print(f"Square root term of m in Z basis (sqrt_term_m_Z): {sqrt_term_m_Z} \n")

# # Vacuum and Single-Photon Events for X and Z basis
# print(f"Vacuum events in X basis (S_X_0_values_list): {min(S_X_0_values_list)} to {max(S_X_0_values_list)}")
# print(f"Single-photon events in X basis (S_X_1_values_list): {min(S_X_1_values_list)} to {max(S_X_1_values_list)}")
# print(f"Vacuum events in Z basis (S_Z_0_values_list): {min(S_Z_0_values_list)} to {max(S_Z_0_values_list)}")
# print(f"Single-photon events in Z basis (S_Z_1_values_list): {min(S_Z_1_values_list)} to {max(S_Z_1_values_list)}\n")

# # Intensity-Level-Specific Parameters
# for i, (mu, P_mu, D_mu, e_mu, n_X_mu, n_Z_mu) in enumerate(
#         zip(mu_k_values, p_mu_k_values, D_mu_k_values, e_mu_k_values, 
#             [n_X_mu_1, n_X_mu_2, n_X_mu_3], [n_Z_mu_1, n_Z_mu_2, n_Z_mu_3]), 1):
#     print(f"\nIntensity Level {i}: μ_{i} = {mu}")
#     print(f"Probability for μ_{i} (P_μ_{i}): {P_mu}")
#     print(f"Detection probability (D_μ_{i}): {D_mu}")
#     print(f"Error rate (e_μ_{i}): {e_mu}")
#     print(f"Events in X basis (n_X_μ_{i}): {n_X_mu}")
#     print(f"Events in Z basis (n_Z_μ_{i}): {n_Z_mu}\n")
    
# # Error bounds in X and Z basis by intensity
# for i, (m_X_mu, m_Z_mu, m_plus_X, m_minus_X, m_plus_Z, m_minus_Z) in enumerate(
#         zip([m_X_mu_1, m_X_mu_2, m_X_mu_3], [m_Z_mu_1, m_Z_mu_2, m_Z_mu_3],
#             [m_plus_X_mu_1, m_plus_X_mu_2, m_plus_X_mu_3], [m_minus_X_mu_1, m_minus_X_mu_2, m_minus_X_mu_3],
#             [m_plus_Z_mu_1, m_plus_Z_mu_2, m_plus_Z_mu_3], [m_minus_Z_mu_1, m_minus_Z_mu_2, m_minus_Z_mu_3]), 1):
#     print(f"Errors in X basis (m_X_μ_{i}): {m_X_mu}")
#     print(f"Bounds in X basis (m_plus_X_μ_{i}, m_minus_X_μ_{i}): ({m_plus_X}, {m_minus_X})")
#     print(f"Errors in Z basis (m_Z_μ_{i}): {m_Z_mu}")
#     print(f"Bounds in Z basis (m_plus_Z_μ_{i}, m_minus_Z_μ_{i}): ({m_plus_Z}, {m_minus_Z})\n") 

# # Single-photon parameters
# print(f"Single-photon events in Z basis (v_Z_1_values): {v_Z_1_values}")
# print(f"Ratio of v_Z_1 to S_Z_1 (v_Z_1_values / S_Z_1_values): {v_Z_1_values / S_Z_1_values}")

# # Security and Key Rate Calculations
# print(f"Observed error rate in X basis (e_obs_X_values): {e_obs_X_values}")
# print(f"Error correction efficiency (lambda_EC_values): {lambda_EC_values}\n")

# print(f"Gamma (security adjustment) (gamma_results): {gamma_results}")
# print(f"Phi_X (binary entropy parameter) (Phi_X_values): {Phi_X_values}")
# print(f"Binary entropy value (binary_entropy_Phi_values): {binary_entropy_Phi_values}")
# print(f"Secret key length (l_calculated_values): {l_calculated_values}")
# print(f"Secret key rate per pulse (R) (key_rates_list): {min(key_rates_list)} to {max(key_rates_list)}")

# print(f"Vacuum events in X basis (S_X_0_values_list): {min(S_X_0_values_list)} to {max(S_X_0_values_list)}")
# print(f"Single-photon events in X basis (S_X_1_values_list): {min(S_X_1_values_list)} to {max(S_X_1_values_list)}")

# %%



