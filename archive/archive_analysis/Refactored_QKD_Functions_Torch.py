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

# def to_device(data, dtype=torch.float32, device=device, device=device):
#     """
#     Converts input data to a tensor on the specified device and dtype.

#     Parameters:
#     - data (Any): Input data to convert (list, float, tensor, etc.).
#     - dtype (torch.dtype): Desired data type of the tensor.
#     - device (str): Target device ('cpu', 'cuda', 'mps').

#     Returns:
#     - torch.Tensor: Converted tensor.
#     """
#     if isinstance(data, torch.Tensor):
#         return data.clone().detach().to(dtype=dtype, device=device)
#     return torch.tensor(data, dtype=dtype, device=device)

def to_device(tensor, device):
    return tensor.to(device) if isinstance(tensor, torch.Tensor) else torch.tensor(tensor, device=device)

# def experimental_parameters():
#     """
#     Define and return experimental parameters for a QKD setup.

#     Returns:
#     - dict: A dictionary containing experimental parameters.
#     """
#     # 1. Fiber lengths and corresponding parameter
#     Ls = torch.linspace(0.1, 200, 1000, device=device, dtype=torch.float32, device=device)  # Fiber lengths in km
#     L_BC = Ls  # Fiber lengths (L_BC)
#     e_1 = L_BC / 100  # Parameter related to fiber length

#     # 2. Dark count probability
#     P_dc_value = torch.tensor(6e-7, device=device, dtype=torch.float32, device=device)  # Dark count probability
#     Y_0 = P_dc_value
#     e_2 = -torch.log(Y_0)  # Related parameter (negative log of Y_0)

#     # 3. Misalignment error
#     e_mis = torch.tensor(5e-3, device=device, dtype=torch.float32, device=device)  # Misalignment error probability
#     e_d = e_mis
#     e_3 = e_d * 100  # Parameter related to misalignment error

#     # # 4. Detected events
#     # n_X_values = [10**s for s in range(6, 11)]  # Detected events (log-scale)
#     # N = n_X_values
#     # e_4 = N  # Parameter related to detected events

#     # 4. Detected events
#     n_X_values = torch.logspace(6, 10, steps=5, device=device, dtype=torch.float32, device=device)  # Detected events (log-scale)
#     N = n_X_values  # Parameter related to detected events
#     e_4 = N

#     # Return parameters as a dictionary
#     return {
#         "fiber_lengths_km": Ls,
#         "e_1": e_1,
#         "dark_count_probability": P_dc_value,
#         "Y_0": Y_0,
#         "e_2": e_2,
#         "misalignment_error_probability": e_mis,
#         "e_d": e_d,
#         "e_3": e_3,
#         "detected_events": n_X_values,
#         "e_4": e_4,
#     }

def experimental_parameters(device=device):
    """
    Define and return experimental parameters for a QKD setup.

    Returns:
    - dict: A dictionary containing experimental parameters.
    """
    # Fiber lengths and related parameters
    Ls = to_device(torch.linspace(0.1, 200, 1000), device=device)  # Fiber lengths
    e_1 = Ls / 100

    # Dark count probability
    P_dc_value = to_device(6e-7, device=device)
    e_2 = -torch.log(P_dc_value)

    # Misalignment error
    e_mis = to_device(5e-3, device=device)
    e_3 = e_mis * 100

    # Detected events
    n_X_values = to_device(torch.logspace(6, 10, steps=5), device=device)
    e_4 = n_X_values

    return {
        "fiber_lengths_km": Ls,
        "e_1": e_1,
        "dark_count_probability": P_dc_value,
        "e_2": e_2,
        "misalignment_error_probability": e_mis,
        "e_3": e_3,
        "detected_events": n_X_values,
        "e_4": e_4,
    }

# Prepare input combinations
# inputs = [(L, n_X) for L in np.linspace(0.1, 200, 100) for n_X in n_X_values]

def other_parameters(device=device):
    # other parameters
    alpha = torch.tensor(0.2, device=device, dtype=torch.float32, device=device)  # Attenuation coefficient (dB/km)
    eta_Bob = torch.tensor(0.1, device=device, dtype=torch.float32, device=device)  # Detector efficiency
    P_ap = torch.tensor(1e-6, device=device, dtype=torch.float32, device=device)  # After-pulse probability
    f_EC = torch.tensor(1.16, device=device, dtype=torch.float32, device=device)  # Error correction efficiency
    epsilon_sec = torch.tensor(1e-10, device=device, dtype=torch.float32, device=device)  # Security error
    epsilon_cor = torch.tensor(1e-15, device=device, dtype=torch.float32, device=device)  # Correlation error
    n_event = torch.tensor(1, device=device, dtype=torch.float32, device=device)  # For single photon event
    kappa = torch.tensor(1e-15, device=device, dtype=torch.float32, device=device)  # Security parameter
    # Return parameters as a dictionary
    return {
        "alpha": alpha,
        "eta_Bob": eta_Bob,
        "P_ap": P_ap,
        "f_EC": f_EC,
        "epsilon_sec": epsilon_sec,
        "epsilon_cor": epsilon_cor,
        "n_event": n_event,
        "kappa": kappa,
    }

def calculate_factorial(n):
    """
    Calculate the factorial using the gamma function for JAX compatibility.
    Factorial of n is gamma(n + 1).
    """
    return torch.lgamma(torch.tensor(n + 1.0, dtype=torch.float32, device=device)).exp()

# def calculate_eta_ch(L, alpha):
#     """
#     Calculates the channel transmittance, eta_ch, based on the fiber length and attenuation coefficient.

#     Parameters:
#     - L (float): Fiber length in kilometers. Expected range: L > 0.
#     - alpha (float): Attenuation coefficient in dB/km. Typical values range from 0.1 to 1 for optical fibers.

#     Returns:
#     - float: Calculated eta_ch value. Restricted to the range [10^-6, 1]. 
#       Returns None if the result is outside this range.
#     """
#     eta = 10 ** (-alpha * L / 10)
#     # return torch.clamp(torch.tensor(eta, dtype=torch.float32, device=device), 1e-6, 1.0)
#     return torch.clamp(eta.clone().detach().to(torch.float32), 1e-6, 1.0)

def calculate_eta_ch(L, alpha, device=device):
    """
    Calculate channel transmittance, eta_ch.

    Parameters:
    - L (float): Fiber length in kilometers.
    - alpha (float): Attenuation coefficient in dB/km.
    - device (str): Target device.

    Returns:
    - torch.Tensor: Calculated eta_ch value.
    """
    L = to_device(L, device=device)
    alpha = to_device(alpha, device=device)
    eta = 10 ** (-alpha * L / 10)
    return torch.clamp(eta, 1e-6, 1.0)

def calculate_eta_sys(eta_Bob, eta_ch):
    """
    Calculates the system transmittance.

    Parameters:
    - eta_Bob (float): Detector efficiency.
    - eta_ch (float): Channel transmittance.

    Returns:
    - float: System transmittance.
    """
    return eta_Bob * eta_ch

# def calculate_D_mu_k(mu_k, eta_sys_values, P_dc):
#     """
#     Calculates detection probability for each intensity level.

#     Parameters:
#     - mu_k (float): Mean photon number for intensity level.
#     - eta_sys (float): System transmittance.

#     Returns:
#     - float: Detection probability.
#     """
#     return 1 - (1 - 2 * P_dc) * torch.exp(-eta_sys_values * mu_k)

def calculate_D_mu_k(mu_k, eta_sys_values, P_dc):
    """
    Calculates detection probability for each intensity level.

    Parameters:
    - mu_k (torch.Tensor): Mean photon number for the intensity level.
    - eta_sys_values (torch.Tensor): System transmittance values.
    - P_dc (torch.Tensor): Dark count probability.

    Returns:
    - torch.Tensor: Detection probability.
    """
    # Ensure all tensors are on the same device
    device = mu_k.device
    return 1 - (1 - 2 * P_dc.to(device)) * torch.exp(-eta_sys_values.to(device) * mu_k.to(device))

def calculate_n_X_total(n_event, mu_1, mu_2, mu_3, P_mu_1, P_mu_2, P_mu_3, P_dc, eta_sys_value, P_X_value, n_X_values):
    """
    Calculates the probability of detection and the expected number of events in the X basis 
    for different intensity levels.

    Parameters:
    - n_event (int): Number of photon events. Must be a non-negative integer.
    - mu_1, mu_2, mu_3 (float): Mean photon numbers for different intensity levels. Must be positive floats.
    - P_mu_1, P_mu_2, P_mu_3 (float): Probabilities for each intensity level. Sum must be approximately 1.
    - P_dc (float): Dark count probability, typically between 0 and 0.1.
    - eta_sys_value (float): System transmittance including channel and detector efficiencies.
    - P_X_value (float): Probability of choosing the X basis. Expected range: [0, 1].
    - n_X_values (float): Total number of events in the X basis.

    Returns:
    - tuple: Contains:
      - sum_P_det_mu_X (float): Total probability of detection for the X basis.
      - n_X_total (float): Sum of expected events for all intensity levels in the X basis.
      - n_X_mu_1, n_X_mu_2, n_X_mu_3 (float): Expected number of events for each intensity in the X basis.
    """
    # Ensure all inputs are 
    device = mu_1.device if isinstance(mu_1, torch.Tensor) else torch.device("cpu")
    # n_event = torch.tensor(n_event, dtype=torch.float32, device=device, device=device)
    # mu_1 = torch.tensor(mu_1, dtype=torch.float32, device=device, device=device)
    # mu_2 = torch.tensor(mu_2, dtype=torch.float32, device=device, device=device)
    # mu_3 = torch.tensor(mu_3, dtype=torch.float32, device=device, device=device)
    # P_mu_1 = torch.tensor(P_mu_1, dtype=torch.float32, device=device, device=device)
    # P_mu_2 = torch.tensor(P_mu_2, dtype=torch.float32, device=device, device=device)
    # P_mu_3 = torch.tensor(P_mu_3, dtype=torch.float32, device=device, device=device)
    # P_dc = torch.tensor(P_dc, dtype=torch.float32, device=device, device=device)
    # eta_sys_value = torch.tensor(eta_sys_value, dtype=torch.float32, device=device, device=device)
    # P_X_value = torch.tensor(P_X_value, dtype=torch.float32, device=device, device=device)
    # n_X_values = torch.tensor(n_X_values, dtype=torch.float32, device=device, device=device)
    
    n_event = to_device(n_event, device=device)
    mu_1 = to_device(mu_1, device=device)
    mu_2 = to_device(mu_2, device=device)
    mu_3 = to_device(mu_3, device=device)
    P_mu_1 = to_device(P_mu_1, device=device)
    P_mu_2 = to_device(P_mu_2, device=device)
    P_mu_3 = to_device(P_mu_3, device=device)
    P_dc = to_device(P_dc, device=device)
    eta_sys_value = to_device(eta_sys_value, device=device)
    P_X_value = to_device(P_X_value, device=device)
    n_X_values = to_device(n_X_values, device=device)
    
    # Calculate the Poisson probabilities for each mu
    P_n_given_mu_1 = (mu_1**n_event * torch.exp(-mu_1)) / calculate_factorial(n_event)
    P_n_given_mu_2 = (mu_2**n_event * torch.exp(-mu_2)) / calculate_factorial(n_event)
    P_n_given_mu_3 = (mu_3**n_event * torch.exp(-mu_3)) / calculate_factorial(n_event)

    # Calculate detection probabilities for each mu under channel condition
    D_mu_1 = 1 - torch.exp(-mu_1 * eta_sys_value) + 2 * P_dc * torch.exp(-mu_1 * eta_sys_value)
    D_mu_2 = 1 - torch.exp(-mu_2 * eta_sys_value) + 2 * P_dc * torch.exp(-mu_2 * eta_sys_value)
    D_mu_3 = 1 - torch.exp(-mu_3 * eta_sys_value) + 2 * P_dc * torch.exp(-mu_3 * eta_sys_value)

    # Calculate joint detection probabilities (detection conditional on mu * chosen probability P_mu_k)
    P_det_mu_1 = D_mu_1 * P_mu_1
    P_det_mu_2 = D_mu_2 * P_mu_2
    P_det_mu_3 = D_mu_3 * P_mu_3

    # Calculate detection probabilities in the X basis (multiply by P_X^2)
    P_det_mu_1_X = P_det_mu_1 * P_X_value**2
    P_det_mu_2_X = P_det_mu_2 * P_X_value**2
    P_det_mu_3_X = P_det_mu_3 * P_X_value**2

    # Total probability of detection on X basis
    sum_P_det_mu_X = P_det_mu_1_X + P_det_mu_2_X + P_det_mu_3_X

    # Conditional probabilities given detection and X basis
    P_mu_1_cond_det_X = P_det_mu_1_X / sum_P_det_mu_X
    P_mu_2_cond_det_X = P_det_mu_2_X / sum_P_det_mu_X
    P_mu_3_cond_det_X = P_det_mu_3_X / sum_P_det_mu_X

    # Expected number of events for each intensity in the X basis
    n_X_mu_1 = n_X_values * P_mu_1_cond_det_X
    n_X_mu_2 = n_X_values * P_mu_2_cond_det_X
    n_X_mu_3 = n_X_values * P_mu_3_cond_det_X
    n_X_total = n_X_mu_1 + n_X_mu_2 + n_X_mu_3

    return sum_P_det_mu_X, P_det_mu_1, P_det_mu_2, P_det_mu_3, n_X_total, n_X_mu_1, n_X_mu_2, n_X_mu_3

def calculate_N(n_X, p_mu_k_values, D_mu_k_values, P_X_value):
    """
    Calculates the expected number of total pulses sent, based on detection rates.

    Parameters:
    - n_X (float): Number of detected events in the X basis.
    - p_mu_k_values (list): Probability values for each intensity.
    - D_mu_k_values (list): Detection probabilities for each intensity.
    - P (float): Basis probability (X or Z).

    Returns:
    - float: Estimated total number of pulses sent.
    """
    terms = torch.tensor([p_mu_k * D_mu_k * P_X_value**2 for p_mu_k, D_mu_k in zip(p_mu_k_values, D_mu_k_values)])
    N_value = n_X / torch.sum(terms)
    return N_value

    # N_value = n_X / sum(p_mu_k * D_mu_k * P_X_value**2 for p_mu_k, D_mu_k in zip(p_mu_k_values, D_mu_k_values))
    # return N_value
    
def calculate_n_Z_total(N_value, p_mu_k_values, D_mu_k_values, P_Z_value, P_det_mu_values):
    """
    Calculates both the total and individual expected number of events in the Z basis for each intensity level.

    Parameters:
    - N_value (float): Total pulses sent.
    - p_mu_k_values (list): Probability values for each intensity.
    - D_mu_k_values (list): Detection probabilities for each intensity.
    - P_Z_value (float): Basis probability for Z basis.
    - P_det_mu_values (list): Detection probabilities for each intensity.

    Returns:
    - tuple: (sum_P_det_mu_Z, n_Z_total, n_Z_mu_1, n_Z_mu_2, n_Z_mu_3)
    """
    # Calculate individual n_Z_mu values based on N_value, intensity probabilities, and Z basis probability
    n_Z_mu_values = torch.tensor([N_value * D_mu_k * p_mu_k * (P_Z_value)**2 for p_mu_k, D_mu_k in zip(p_mu_k_values, D_mu_k_values)])
    n_Z_total = torch.sum(n_Z_mu_values)
    # Expected number of events for each intensity level in the Z basis using conditional probabilities
    # n_Z_mu_values = [n_Z_mu_total * P_mu_cond_det_Z for P_mu_cond_det_Z in P_mu_cond_det_Z_values]
    # n_Z_total = jnp.sum(n_Z_mu_values)

    # Calculate detection probabilities in the Z basis (multiply by P_Z^2 for each)
    P_det_mu_Z_values = torch.tensor([P_det_mu * (P_Z_value)**2 for P_det_mu in P_det_mu_values])
    sum_P_det_mu_Z = torch.sum(P_det_mu_Z_values)

    # Calculate conditional probabilities given detection and Z basis
    # P_mu_cond_det_Z_values = [P_det_mu_Z / sum_P_det_mu_Z for P_det_mu_Z in P_det_mu_Z_values]

    # Unpack for individual returns
    n_Z_mu_1, n_Z_mu_2, n_Z_mu_3 = n_Z_mu_values

    return sum_P_det_mu_Z, n_Z_total, n_Z_mu_1, n_Z_mu_2, n_Z_mu_3

def calculate_e_mu_k(P_dc, e_mis, P_ap, D_mu_k, eta_sys_values, mu_k):
    """
    Calculates the error rate for a single intensity level.

    Parameters:
    - P_dc (float): Dark count probability.
    - e_mis (float): Misalignment error probability.
    - P_ap (float): After pulse probability.
    - D_mu_k (float): Detection probability for the intensity level.
    - eta_sys (float): System transmittance.
    - mu_k (float): Mean photon number for intensity level.

    Returns:
    - float: Error rate for the intensity level.
    """
    return P_dc + e_mis * (1 - torch.exp(-eta_sys_values * mu_k)) + P_ap * D_mu_k / 2

def calculate_e_obs(m_X_total, n_X_values):
    """
    Calculate the observed error rate in a specified basis (X or Z) given the number 
    of events for each intensity and error rates.

    Parameters:
    - n_mu_k_values (list): Number of detected events in the specified basis for each intensity.
    - e_mu_k_values (list): Error rates for each intensity.
    - n_total (float): Total number of events in the specified basis.

    Returns:
    - float: Observed error rate in the specified basis.
    """
    # return sum(n_mu_k / n_total * e_mu_k for n_mu_k, e_mu_k in zip(n_mu_k_values, e_mu_k_values))
    # result = m_X_total / n_X_values

    # return min(result, 0.5)
    result = m_X_total / n_X_values
    return torch.minimum(result, torch.tensor(0.5, dtype=torch.float32, device=device))

def calculate_h(x):
    """
    Binary entropy function.

    Parameters:
    - x (float): Error rate (0 <= x <= 1).

    Returns:
    - float: Binary entropy of x.
    """
    # # if x == 0 or x == 1:
    # #     return 0
    # return -x * jnp.log2(x) - (1 - x) * jnp.log2(1 - x) 
    x = torch.clamp(x, 1e-10, 1 - 1e-10)  # Avoid log(0)
    return -x * torch.log2(x) - (1 - x) * torch.log2(1 - x)

def calculate_lambda_EC(n_X_values, f_EC, e_obs):
    """
    Calculates the error correction term.

    Parameters:
    - n_X_value (float): Total detected events in X basis.
    - f_EC (float): Error correction efficiency.
    - calculate_e_obs (float): Observed error rate.

    Returns:
    - float: Error correction term.
    """
    return n_X_values * f_EC * calculate_h(e_obs)

def calculate_sqrt_term(n_total, epsilon_sec):
    """
    Calculate the square root term used in uncertainty calculations for a given basis.

    Parameters:
    - n_total (float): Total number of events in the specified basis (X or Z).
    - epsilon_sec (float): Security parameter epsilon for secrecy.

    Returns:
    - float: Calculated square root term.
    """
    epsilon_sec = torch.clamp(epsilon_sec, 1e-10, None)  # Avoid log(0)
    return torch.sqrt((n_total / 2) * torch.log(21 / epsilon_sec))


def calculate_tau_n(n, mu_k_values, p_mu_k_values):
    """
    Calculates Poisson probabilities for intensity levels using JAX.

    Parameters:
    - n (int): Photon number.
    - mu_k_values (list or array): Mean photon numbers.
    - p_mu_k_values (list or array): Probability values.

    Returns:
    - float: Probability of detecting n photons.
    """
    mu_k_values = torch.tensor(mu_k_values, dtype=torch.float32, device=device)
    p_mu_k_values = torch.tensor(p_mu_k_values, dtype=torch.float32, device=device)

    # Calculate tau values
    tau_values = p_mu_k_values * torch.exp(-mu_k_values) * (mu_k_values ** n) / torch.exp(torch.lgamma(torch.tensor(n + 1.0)))
    return torch.sum(tau_values)


def calculate_n_pm(mu_k_values, p_mu_k_values, n_mu_k, sqrt_term):
    """
    Calculate the bounds for a specific intensity level.

    Parameters:
    - mu_k_values (Tensor): Intensity levels.
    - p_mu_k_values (Tensor): Probabilities for intensity levels.
    - n_mu_k (Tensor): Detected events for intensity.
    - sqrt_term (Tensor): Uncertainty term.

    Returns:
    - tuple: (n_plus, n_minus).
    """
    # Ensure all inputs are tensors
    device = mu_k_values.device if isinstance(mu_k_values, torch.Tensor) else torch.device("cpu")
    mu_k_values = torch.tensor(mu_k_values, dtype=torch.float32, device=device, device=device)
    p_mu_k_values = torch.tensor(p_mu_k_values, dtype=torch.float32, device=device, device=device)
    n_mu_k = torch.tensor(n_mu_k, dtype=torch.float32, device=device, device=device)
    sqrt_term = torch.tensor(sqrt_term, dtype=torch.float32, device=device, device=device)

    # Compute n_plus and n_minus
    n_plus = ((torch.exp(mu_k_values)) / p_mu_k_values) * (n_mu_k + sqrt_term)
    n_minus = ((torch.exp(mu_k_values)) / p_mu_k_values) * torch.maximum(n_mu_k - sqrt_term, torch.tensor(0.0, dtype=torch.float32, device=device, device=device))
    
    return n_plus, n_minus

# Number of vacuum events
def calculate_S_0(tau_0, mu_2, mu_3, n_minus_mu_3, n_plus_mu_2):
    """
    Calculate S_0 for the basis (X or Z).

    Parameters:
    - tau_0 (float): Poisson probability for 0 photons.
    - mu_2, mu_3 (float): Mean photon numbers.
    - n_minus_mu_3, n_plus_mu_2 (float): Bounds on detected events.

    Returns:
    - float: S_0 for the basis.
    """
    result = tau_0 * (mu_2 * n_minus_mu_3 - mu_3 * n_plus_mu_2) / (mu_2 - mu_3)
    return torch.maximum(result, torch.tensor(0.0, dtype=torch.float32, device=device))


def calculate_S_1(tau_1, mu_1, mu_2, mu_3, n_minus_mu_2, n_plus_mu_3, n_plus_mu_1, s_0, tau_0):
    """
    Calculate S_1 for the basis (X or Z).

    Parameters:
    - tau_1 (float): Poisson probability for 1 photon.
    - mu_1, mu_2, mu_3 (float): Mean photon numbers.
    - n_minus_mu_2, n_plus_mu_3, n_plus_mu_1 (float): Bounds on detected events.
    - s_0 (float): S_0 for the basis.
    - tau_0 (float): Poisson probability for 0 photons.

    Returns:
    - float: S_1 for the basis.
    """
    numerator = tau_1 * mu_1 * (n_minus_mu_2 - n_plus_mu_3 - ((mu_2**2 - mu_3**2) / mu_1**2) * (n_plus_mu_1 - s_0 / tau_0))
    denominator = mu_1 * (mu_2 - mu_3) - mu_2**2 + mu_3**2
    result = numerator / denominator
    return torch.maximum(result, torch.tensor(0.0, dtype=torch.float32, device=device))

def calculate_m_mu_k(e_mu_k_values, p_mu_k_values, N_value, P):
    """
    Calculates m_k for each intensity level in a specified basis.

    Parameters:
    - e_mu_k_values (list of floats): Error rates for each intensity.
    - p_mu_k_values (list of floats): Probabilities for each intensity.
    - N_value (float): Total pulses sent.
    - P (float): Basis probability.

    Returns:
    - list of floats: m_k values for each intensity level.
    """
    # Calculate m_k for each intensity
    m_mu_k_values = [e_mu_k * N_value * (P**2) * p_mu_k for e_mu_k, p_mu_k in zip(e_mu_k_values, p_mu_k_values)]
    return m_mu_k_values

def calculate_m_pm(mu_k_values, p_mu_k_values, m_k, sqrt_term):
    """
    Calculates bounds on m_k for uncertainty.

    Parameters:
    - mu_k_values (Tensor or float): Mean photon numbers.
    - p_mu_k_values (Tensor or float): Probabilities for each intensity.
    - m_k (Tensor or float): Error rate for intensity.
    - sqrt_term (Tensor or float): Uncertainty term.

    Returns:
    - tuple: (m_plus, m_minus).
    """
    # Ensure all inputs are tensors and on the same device
    device = mu_k_values.device if isinstance(mu_k_values, torch.Tensor) else torch.device("cpu")
    mu_k_values = torch.tensor(mu_k_values, dtype=torch.float32, device=device, device=device)
    p_mu_k_values = torch.tensor(p_mu_k_values, dtype=torch.float32, device=device, device=device)
    m_k = torch.tensor(m_k, dtype=torch.float32, device=device, device=device)
    sqrt_term = torch.tensor(sqrt_term, dtype=torch.float32, device=device, device=device)

    # Calculate m_plus and m_minus
    m_plus = (torch.exp(mu_k_values) / p_mu_k_values) * (m_k + sqrt_term)
    m_minus = (torch.exp(mu_k_values) / p_mu_k_values) * torch.maximum(m_k - sqrt_term, torch.tensor(0.0, dtype=torch.float32, device=device, device=device))

    return m_plus, m_minus

def calculate_v_1(tau_1, m_plus_mu_2, m_minus_mu_3, mu_2, mu_3):
    """
    Calculates v_1, a value used in the error rate estimation, for a specified basis (X or Z).

    Parameters:
    - tau_1 (float): Probability of detecting 1 photon.
    - m_plus_mu_2 (float): Upper bound on the number of detected events for the second intensity level.
    - m_minus_mu_3 (float): Lower bound on the number of detected events for the third intensity level.
    - mu_2, mu_3 (float): Mean photon numbers for second and third intensities.

    Returns:
    - float: v_1, the calculated parameter.
    """
    return tau_1 * (m_plus_mu_2 - m_minus_mu_3) / (mu_2 - mu_3)

def calculate_gamma(a, b, c, d):
    """
    Calculates the gamma adjustment term for key rate correction.

    Parameters:
    - a, b, c, d (torch.Tensor): Input parameters for gamma calculation.

    Returns:
    - torch.Tensor: Gamma adjustment term.
    """
    if (b == 0) or (b == 1):
        return torch.tensor(0.0, dtype=torch.float32, device=device)

    term1 = (c + d) * (1 - b) * b / (c * d * torch.log(torch.tensor(2.0)))
    term2 = torch.log2((c + d) / (c * d * (1 - b) * b) * (21**2 / a**2))
    return torch.sqrt(term1 * term2)

def calculate_Phi(v_Z_1, s_Z_1, gamma_result):
    """
    Calculates the Phi term, representing the key rate error correction bound.

    Parameters:
    - v_1 (float): Calculated v_1 parameter from error rate estimations.
    - S_1 (float): Number of single-photon events for the basis.
    - gamma_result (float): Gamma adjustment term.

    Returns:
    - float: Phi value used in the binary entropy calculation.
    """
    result = v_Z_1 / s_Z_1 + gamma_result
    return torch.minimum(result, torch.tensor(0.5, dtype=torch.float32, device=device))

def calculate_LastTwoTerm(epsilon_sec, epsilon_cor):
    """
    Calculates the final adjustment term in the security bound, dependent on the secrecy and correctness parameters.

    Parameters:
    - epsilon_sec (float): Secrecy parameter epsilon.
    - epsilon_cor (float): Correctness parameter epsilon.

    Returns:
    - float: Adjustment term for the key rate calculation.
    """
    return -6 * torch.log2(21 / epsilon_sec) - torch.log2(2 / epsilon_cor)

def calculate_l(S_X_0_values, S_X_1_values, binary_entropy_Phi_values, lambda_EC_values, epsilon_sec, epsilon_cor):
    """
    Calculates the final secret key length for the specified basis.

    Parameters:
    - S_0, S_1 (float): Estimated single-photon event counts for each intensity level.
    - binary_entropy_value_Phi (float): Binary entropy function result for Phi.
    - lambda_EC_value (float): Error correction term.
    - epsilon_sec, epsilon_cor (float): Security parameters for secrecy and correctness.

    Returns:
    - float: Secret key length (l).
    """
    adjustment_term = calculate_LastTwoTerm(epsilon_sec, epsilon_cor)
    l_value = S_X_0_values + S_X_1_values - S_X_1_values * binary_entropy_Phi_values - lambda_EC_values + adjustment_term
    return torch.maximum(l_value, torch.tensor(0.0, dtype=torch.float32, device=device))  # Ensure non-negative key length
    

    # Ensure the key length is non-negative
    

def calculate_R(l: float, N: float) -> float:
    """
    Calculates the secret key rate per pulse.
    
    Parameters:
    - l (float): Secret key length, must be non-negative.
    - N (float): Total number of pulses sent, must be positive.
    
    Returns:
    - float: Secret key rate per pulse (R).

    """
    # assert l >= 0, "Secret key length (l) must be non-negative."
    # assert N > 0, "Total pulses sent (N) must be positive."
    return l / N


def validate_parameters_and_conditions(mu_1, mu_2, mu_3, P_mu_1, P_mu_2, P_mu_3, P_X_value, P_Z_value,
                                       Ls, alpha, P_dc_value, P_ap, eta_sys_values, eta_ch_values,
                                       eta_Bob, epsilon_sec, epsilon_cor, f_EC, n_X_values, n_Z_mu_values, n_Z_total):
    """
    Validates key conditions and parameter ranges for decoy-state QKD setup.

    Returns:
    - bool: True if all conditions and range checks are satisfied, else False.
    """
    # Core conditions for parameter relationships
    conditions = {
        "Condition 1: mu_1 > mu_2 + mu_3": (mu_1 > mu_2 + mu_3).item(),
        "Condition 2: mu_2 > mu_3 >= 0": (mu_2 > mu_3 >= 0).item(),
        "Condition 3: P_mu values sum to 1": torch.abs(P_mu_1 + P_mu_2 + P_mu_3 - 1) < 1e-5,
        "Condition 4: P_X + P_Z sum to 1": torch.abs(P_X_value + P_Z_value - 1) < 1e-5
    }
    
    # Print and evaluate each condition
    all_conditions_passed = True
    for condition, is_satisfied in conditions.items():
        status = "Pass" if is_satisfied else "Fail"
        print(f"{condition}: {status}")
        all_conditions_passed &= is_satisfied  # Update the final status

    # Bound checks for parameters
    bounds = {
        "Minimum Fiber Length": (Ls.min().item(), (0.1, 200)),
        "Maximum Fiber Length": (Ls.max().item(), (0.1, 200)),
        "Attenuation Coefficient (alpha)": (alpha.item(), (0.1, 1)),
        "Dark Count Probability (P_dc)": (P_dc_value.item(), (1e-8, 1e-5)),
        "After Pulse Probability (P_ap)": (P_ap.item(), (0, 0.1)),
        "System Transmittance (eta_sys)": (eta_sys_values.min().item(), eta_sys_values.max().item(), (1e-6, 1)),
        "Channel Transmittance (eta_ch)": (eta_ch_values.min().item(), eta_ch_values.max().item(), (1e-6, 1)),
        "Detector Efficiency (eta_Bob)": (eta_Bob.item(), (0, 1)),
        "Secrecy Parameter (epsilon_sec)": (epsilon_sec.item(), (1e-10, 1)),
        "Correctness Parameter (epsilon_cor)": (epsilon_cor.item(), (1e-15, 1)),
        "Error Correction Efficiency (f_EC)": (f_EC.item(), (1, 2)),
        "Detected Events in X Basis (n_X_values)": (n_X_values.min().item(), n_X_values.max().item(), (1e9, 1e11)),
        "Detected Events in Z Basis (n_Z)": (n_Z_mu_values.min().item(), n_Z_mu_values.max().item(), (1e8, 1e11)),
        "Total Events in Z Basis (n_Z_total)": (n_Z_total.item(), (1e9, 1e11)),
    }

    # Check each parameter's bounds and store out-of-bound values
    all_bounds_passed = True
    out_of_bound_params = []
    for name, (value, range_) in bounds.items():
        if isinstance(value, torch.Tensor):  # Handle tensors
            within_bounds = (range_[0] <= value.item() <= range_[1])
            status = "within bounds" if within_bounds else "out of bounds"
            print(f"{name}: {value.item()} ({status}) - Expected range: {range_}")
            if not within_bounds:
                out_of_bound_params.append((name, value.item(), range_))
                all_bounds_passed = False
        elif isinstance(value, tuple):  # Handle min-max tuple bounds
            min_val, max_val = value
            min_within = range_[0] <= min_val
            max_within = max_val <= range_[1]
            status = "within bounds" if min_within and max_within else "out of bounds"
            print(f"{name}: ({min_val}, {max_val}) ({status}) - Expected range: {range_}")
            if not (min_within and max_within):
                out_of_bound_params.append((name, value, range_))
                all_bounds_passed = False

    # Print out-of-bound values, if any
    if out_of_bound_params:
        print("\nThe following parameters are out of bounds:")
        for param in out_of_bound_params:
            name, value, range_ = param
            print(f"  - {name}: {value} (Expected range: {range_})")

    # Final validation status
    if all_conditions_passed and all_bounds_passed:
        print("\n✅ All conditions and parameter ranges are within expected bounds.")
        return True
    else:
        print("\n❌ One or more conditions/parameters are out of bounds.")
        return False

# Define the `objective` function with `alpha` and other parameters as arguments
def calculate_key_rates_and_metrics(params, L_values, n_X, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event, device): 
#  mu_k_values, eta_ch_values, p_mu_k_values, p_1 = mu_1, p_2 = mu_2, p_3 = P_mu_1 = 0.65, p_4 = P_mu_2 = 0.3, p_5 =        P_X_value = 5e-3
    mu_1, mu_2, P_mu_1, P_mu_2, P_X_value = params 
    mu_3 = torch.tensor(2e-4, dtype=torch.float32, device=device)
    mu_k_values = torch.tensor([mu_1, mu_2, mu_3], dtype=torch.float32, device=device)
    P_mu_3 = 1 - P_mu_1 - P_mu_2
    p_mu_k_values = torch.tensor([P_mu_1, P_mu_2, P_mu_3], dtype=torch.float32, device=device)
    P_Z_value = 1 - P_X_value
       
    # n_X = jnp.array([10**s for s in range(6, 11)])  # Detected events in X basis: 10^6 to 10^10
    """Objective function to optimize key rate."""
# 1. Channel and system efficiencies
    eta_ch_values = calculate_eta_ch(L_values, alpha, device=device)  # Channel transmittance
    eta_sys_values = calculate_eta_sys(eta_Bob.to(device), eta_ch_values.to(device))  # System transmittance

    # 2. Detection probabilities for each intensity level
    # D_mu_k_values = torch.tensor([calculate_D_mu_k(mu_k, eta_sys_values, P_dc_value) for mu_k in mu_k_values], dtype=torch.float32, device=device)
    D_mu_k_values = [calculate_D_mu_k(mu_k.to(device), eta_sys_values.to(device), P_dc_value.to(device))
                 for mu_k in mu_k_values]
    D_mu_k_values = torch.stack(D_mu_k_values).to(device)
    # 3. Error rates for each intensity level
    e_mu_k_values = torch.tensor([calculate_e_mu_k(P_dc_value, e_mis, P_ap, D_mu_k, eta_sys_values, mu_k)
                                   for D_mu_k, mu_k in zip(D_mu_k_values, mu_k_values)], dtype=torch.float32, device=device)

    # 4. Detection probabilities and events in the X basis
    sum_P_det_mu_X, P_det_mu_1, P_det_mu_2, P_det_mu_3, n_X_total, n_X_mu_1, n_X_mu_2, n_X_mu_3 = calculate_n_X_total(
        n_event, mu_1, mu_2, mu_3, P_mu_1, P_mu_2, P_mu_3, P_dc_value, eta_sys_values, P_X_value, n_X
    )
    sqrt_term_n_X = calculate_sqrt_term(n_X, epsilon_sec)

    n_X_mu_k_values = torch.tensor([n_X_mu_1, n_X_mu_2, n_X_mu_3], dtype=torch.float32, device=device)
    n_X_total = torch.sum(n_X_mu_k_values)

    P_det_mu_values = [P_det_mu_1, P_det_mu_2, P_det_mu_3]

    n_plus_X_mu_1, n_minus_X_mu_1 = calculate_n_pm(mu_1, P_mu_1, n_X_mu_1, sqrt_term_n_X)
    n_plus_X_mu_2, n_minus_X_mu_2 = calculate_n_pm(mu_2, P_mu_2, n_X_mu_2, sqrt_term_n_X)
    n_plus_X_mu_3, n_minus_X_mu_3 = calculate_n_pm(mu_3, P_mu_3, n_X_mu_3, sqrt_term_n_X)


    # 5. Total pulses and events in Z basis
    N_values = calculate_N(n_X_total, p_mu_k_values, D_mu_k_values, P_X_value)
    sum_P_det_mu_Z, n_Z_total, n_Z_mu_1, n_Z_mu_2, n_Z_mu_3 = calculate_n_Z_total(
        N_values, p_mu_k_values, D_mu_k_values, P_Z_value, P_det_mu_values
    )
    sqrt_term_n_Z = calculate_sqrt_term(n_Z_total, epsilon_sec)

    n_Z_mu_values = torch.tensor([n_Z_mu_1, n_Z_mu_2, n_Z_mu_3], dtype=torch.float32, device=device)

    n_plus_Z_mu_1, n_minus_Z_mu_1 = calculate_n_pm(mu_1, P_mu_1, n_Z_mu_1, sqrt_term_n_Z)
    n_plus_Z_mu_2, n_minus_Z_mu_2 = calculate_n_pm(mu_2, P_mu_2, n_Z_mu_2, sqrt_term_n_Z)
    n_plus_Z_mu_3, n_minus_Z_mu_3 = calculate_n_pm(mu_3, P_mu_3, n_Z_mu_3, sqrt_term_n_Z)

    # 7. Security-related terms
    tau_0_values = calculate_tau_n(0, mu_k_values, p_mu_k_values)
    tau_1_values = calculate_tau_n(1, mu_k_values, p_mu_k_values)

    # 8. Error terms for X basis
    m_X_mu_values = calculate_m_mu_k(e_mu_k_values, p_mu_k_values, N_values, P_X_value)
    m_X_mu_values = torch.tensor(m_X_mu_values, dtype=torch.float32, device=device)

    m_X_total = torch.sum(m_X_mu_values)
    sqrt_term_m_X = calculate_sqrt_term(m_X_total, epsilon_sec)

    m_plus_X_mu_1, m_minus_X_mu_1 = calculate_m_pm(mu_1, P_mu_1, m_X_mu_values[0], sqrt_term_m_X)
    m_plus_X_mu_2, m_minus_X_mu_2 = calculate_m_pm(mu_2, P_mu_2, m_X_mu_values[1], sqrt_term_m_X)
    m_plus_X_mu_3, m_minus_X_mu_3 = calculate_m_pm(mu_3, P_mu_3, m_X_mu_values[2], sqrt_term_m_X)

    e_obs_X_values = calculate_e_obs(m_X_total, n_X)

    # 9. Error terms for Z basis
    m_Z_mu_values = calculate_m_mu_k(e_mu_k_values, p_mu_k_values, N_values, P_Z_value)
    m_Z_total = torch.sum(torch.tensor(m_Z_mu_values, dtype=torch.float32, device=device))
    sqrt_term_m_Z = calculate_sqrt_term(m_Z_total, epsilon_sec)

    m_plus_Z_mu_1, m_minus_Z_mu_1 = calculate_m_pm(mu_1, P_mu_1, m_Z_mu_values[0], sqrt_term_m_Z)
    m_plus_Z_mu_2, m_minus_Z_mu_2 = calculate_m_pm(mu_2, P_mu_2, m_Z_mu_values[1], sqrt_term_m_Z)
    m_plus_Z_mu_3, m_minus_Z_mu_3 = calculate_m_pm(mu_3, P_mu_3, m_Z_mu_values[2], sqrt_term_m_Z)

    # 10. Contributions for single-photon events
    S_X_0_values = calculate_S_0(tau_0_values, mu_2, mu_3, n_minus_X_mu_3, n_plus_X_mu_2)
    S_Z_0_values = calculate_S_0(tau_0_values, mu_2, mu_3, n_minus_Z_mu_3, n_plus_Z_mu_2)

    S_X_1_values = calculate_S_1(tau_1_values, mu_1, mu_2, mu_3, n_minus_X_mu_2, n_plus_X_mu_3, n_plus_X_mu_1, S_X_0_values, tau_0_values)
    S_Z_1_values = calculate_S_1(tau_1_values, mu_1, mu_2, mu_3, n_minus_Z_mu_2, n_plus_Z_mu_3, n_plus_Z_mu_1, S_Z_0_values, tau_0_values)

    # 11. Security bounds and key length
    v_Z_1_values = calculate_v_1(tau_1_values, m_plus_Z_mu_2, m_minus_Z_mu_3, mu_2, mu_3)

    gamma_results = calculate_gamma(epsilon_sec, v_Z_1_values / (S_Z_1_values + 1e-10), S_Z_1_values, S_X_1_values)
    Phi_X_values = calculate_Phi(v_Z_1_values, S_Z_1_values, gamma_results)
    binary_entropy_Phi_values = calculate_h(Phi_X_values)

    lambda_EC_values = calculate_lambda_EC(n_X, f_EC, e_obs_X_values)
    l_calculated_values = calculate_l(S_X_0_values, S_X_1_values, binary_entropy_Phi_values, lambda_EC_values, epsilon_sec, epsilon_cor)

    key_rates = calculate_R(l_calculated_values, N_values)

    return (
        key_rates,
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
        l_calculated_values, device
    )

def penalty(key_rates, mu_1, mu_2, mu_3, P_mu_1, P_mu_2, P_mu_3):
    """Penalty function to enforce constraints."""
    # Ensure all inputs are tensors
    device = mu_1.device if isinstance(mu_1, torch.Tensor) else torch.device("cpu")
    mu_1 = torch.tensor(mu_1, dtype=torch.float32, device=device, device=device)
    mu_2 = torch.tensor(mu_2, dtype=torch.float32, device=device, device=device)
    mu_3 = torch.tensor(mu_3, dtype=torch.float32, device=device, device=device)
    P_mu_1 = torch.tensor(P_mu_1, dtype=torch.float32, device=device, device=device)
    P_mu_2 = torch.tensor(P_mu_2, dtype=torch.float32, device=device, device=device)
    P_mu_3 = torch.tensor(P_mu_3, dtype=torch.float32, device=device, device=device)

    # Compute penalties
    penalty_mu1_sum = torch.where(mu_1 > mu_2 + mu_3, torch.tensor(0.0, device=device), torch.tensor(1e6, device=device))
    penalty_mu2_ratio = torch.where(mu_2 / mu_1 < 1, torch.tensor(0.0, device=device), torch.tensor(1e6, device=device))
    penalty_sum = torch.where(torch.abs(P_mu_1 + P_mu_2 + P_mu_3 - 1) < 1e-6, torch.tensor(0.0, device=device), torch.tensor(1e6, device=device))
    penalty_P_mu_3 = torch.where(P_mu_3 > 0, torch.tensor(0.0, device=device), torch.tensor(1e6, device=device))
    penalty_mu2_mu3 = torch.where(mu_2 > mu_3, torch.tensor(0.0, device=device), torch.tensor(1e6, device=device))

    # Combine all penalties
    total_penalty = penalty_mu1_sum + penalty_mu2_ratio + penalty_sum + penalty_mu2_mu3 + penalty_P_mu_3

    # Penalized key rates
    penalized_key_rates = key_rates - total_penalty
    return penalized_key_rates

def objective(params, L_values, n_X, alpha, eta_Bob, P_dc, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event, device=device):
    """
    Objective function for QKD key rate optimization.

    Parameters:
    - params: Optimization parameters.
    - device (str): Target device.

    Returns:
    - Tuple: Penalized key rates and metrics.
    """
    mu_1, mu_2, P_mu_1, P_mu_2, P_X = params
    mu_3 = to_device(2e-4, device=device)
    P_mu_3 = 1 - P_mu_1 - P_mu_2
    p_mu_k = [P_mu_1, P_mu_2, P_mu_3]

    # Metrics
    key_rates, eta_ch, *_ = calculate_key_rates_and_metrics(
        params, L_values, n_X, alpha, eta_Bob, P_dc, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event, device
    )

    # Penalized key rates
    penalized_key_rates = penalty(key_rates, mu_1, mu_2, mu_3, P_mu_1, P_mu_2, P_mu_3)
    return penalized_key_rates
    
# def objective(params, L_values, n_X, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event):
#     """
#     Objective function with penalty applied to key rates.
#     """
#     # Unpack parameters
#     mu_1, mu_2, P_mu_1, P_mu_2, P_X_value = params
#     mu_3 = 2e-4  # Ensure mu_3 is defined
#     P_mu_3 = 1 - P_mu_1 - P_mu_2  # Derived value for P_mu_3
    
#     # Compute metrics (simulate)
#     (
#         key_rates,
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
#         l_calculated_values
#     ) = calculate_key_rates_and_metrics(params, L_values, n_X, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event)

    
#     # Apply penalty to key rates
#     penalized_key_rates = penalty(key_rates, mu_1, mu_2, mu_3, P_mu_1, P_mu_2, P_mu_3)
    
#     # Return all metrics including penalized key rates
#     return (
#         penalized_key_rates,
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
#         l_calculated_values
#     )
