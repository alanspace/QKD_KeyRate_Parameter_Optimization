from math import exp, factorial

# The use of jax.config.update("jax_enable_x64", True) ensures double-precision floating-point operations, which can improve numerical stability for quantum key distribution (QKD) calculations.
import jax
jax.config.update("jax_enable_x64", True)
from jax import grad, jit, vmap
import jax.numpy as jnp
from jax.scipy.special import logsumexp
# This import is appropriate for factorial calculations in JAX.
from jax.scipy.special import gamma

# dual_annealing is a SciPy optimization function that does not directly support JAX arrays (jax.numpy).
from scipy.optimize import minimize, dual_annealing, differential_evolution
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
import functools
from joblib import Parallel, delayed
import os
# Import necessary libraries
import json
from concurrent.futures import ProcessPoolExecutor
import time
import numpy as np
import jax.numpy as jnp
import pandas as pd
from tabulate import tabulate
import json
from jax import lax

def experimental_parameters():
    """
    Define and return experimental parameters for a QKD setup.

    Returns:
    - dict: A dictionary containing experimental parameters.
    """
    # 1. Fiber lengths and corresponding parameter
    Ls = np.linspace(0.1, 200, 1000)  # Fiber lengths in km
    L_BC = Ls  # Fiber lengths (L_BC)
    e_1 = L_BC / 100  # Parameter related to fiber length

    # 2. Dark count probability
    P_dc_value = 6 * 10**-7  # Dark count probability
    Y_0 = P_dc_value
    e_2 = -jnp.log(Y_0)  # Related parameter (negative log of Y_0)

    # 3. Misalignment error
    e_mis = 5 * 10**-3  # Misalignment error probability
    e_d = e_mis
    e_3 = e_d * 100  # Parameter related to misalignment error

    # 4. Detected events
    n_X_values = [10**s for s in range(6, 11)]  # Detected events (log-scale)
    N = n_X_values
    e_4 = N  # Parameter related to detected events

    # Return parameters as a dictionary
    return {
        "fiber_lengths_km": Ls,
        "e_1": e_1,
        "dark_count_probability": P_dc_value,
        "Y_0": Y_0,
        "e_2": e_2,
        "misalignment_error_probability": e_mis,
        "e_d": e_d,
        "e_3": e_3,
        "detected_events": n_X_values,
        "e_4": e_4,
    }

# Prepare input combinations
# inputs = [(L, n_X) for L in np.linspace(0.1, 200, 100) for n_X in n_X_values]

def other_parameters():
    # other parameters
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
    P_ap = 0
    # 4*1e-2          # given in the paper, discussed with range from 0 to 0.1
    kappa = 1e-15           # given in the paper
    f_EC = 1.16             # given in the paper, range around 1.1
    return

def calculate_factorial(n):
    """
    Calculate the factorial using the gamma function for JAX compatibility.
    Factorial of n is gamma(n + 1).
    """
    return gamma(n + 1)

def calculate_eta_ch(L, alpha):
    """
    Calculates the channel transmittance, eta_ch, based on the fiber length and attenuation coefficient.

    Parameters:
    - L (float): Fiber length in kilometers. Expected range: L > 0.
    - alpha (float): Attenuation coefficient in dB/km. Typical values range from 0.1 to 1 for optical fibers.

    Returns:
    - float: Calculated eta_ch value. Restricted to the range [10^-6, 1]. 
      Returns None if the result is outside this range.
    """
    eta = 10 ** (-alpha * L / 10)
    return eta 

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

def calculate_D_mu_k(mu_k, eta_sys_values, P_dc):
    """
    Calculates detection probability for each intensity level.

    Parameters:
    - mu_k (float): Mean photon number for intensity level.
    - eta_sys (float): System transmittance.

    Returns:
    - float: Detection probability.
    """
    return 1 - (1 - 2 * P_dc) * jnp.exp(-eta_sys_values * mu_k)

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
    - P (float): Probability of choosing the X basis. Expected range: [0, 1].
    - n_X_values (float): Total number of events in the X basis.

    Returns:
    - tuple: Contains:
      - sum_P_det_mu_X (float): Total probability of detection for the X basis.
      - n_X_total (float): Sum of expected events for all intensity levels in the X basis.
      - n_X_mu_1, n_X_mu_2, n_X_mu_3 (float): Expected number of events for each intensity in the X basis.
    """
     
    # Calculate the Poisson probabilities for each mu
    P_n_given_mu_1 = (mu_1**n_event * jnp.exp(-mu_1))/ calculate_factorial(n_event)
    P_n_given_mu_2 = (mu_2**n_event * jnp.exp(-mu_2)) / calculate_factorial(n_event)
    P_n_given_mu_3 = (mu_3**n_event * jnp.exp(-mu_3)) / calculate_factorial(n_event)
    
    # Calculate detection probabilities for each mu under channel condition
    D_mu_1 = 1 - jnp.exp(-mu_1* eta_sys_value) + 2 * P_dc * jnp.exp(-mu_1 * eta_sys_value) # P_det_cond_mu_1
    D_mu_2 = 1 - jnp.exp(-mu_2* eta_sys_value) + 2 * P_dc * jnp.exp(-mu_2 * eta_sys_value) # P_det_cond_mu_2
    D_mu_3 = 1 - jnp.exp(-mu_3* eta_sys_value) + 2 * P_dc * jnp.exp(-mu_3 * eta_sys_value) # P_det_cond_mu_3
    
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
    terms = jnp.array([p_mu_k * D_mu_k * P_X_value**2 for p_mu_k, D_mu_k in zip(p_mu_k_values, D_mu_k_values)])
    N_value = n_X / jnp.sum(terms)
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
    n_Z_mu_values = jnp.array([N_value * D_mu_k * p_mu_k * (P_Z_value)**2 for p_mu_k, D_mu_k in zip(p_mu_k_values, D_mu_k_values)])
    # Expected number of events for each intensity level in the Z basis using conditional probabilities
    # n_Z_mu_values = [n_Z_mu_total * P_mu_cond_det_Z for P_mu_cond_det_Z in P_mu_cond_det_Z_values]
    n_Z_total = jnp.sum(n_Z_mu_values)

    # Calculate detection probabilities in the Z basis (multiply by P_Z^2 for each)
    P_det_mu_Z_values = jnp.array([P_det_mu * (P_Z_value)**2 for P_det_mu in P_det_mu_values])
    sum_P_det_mu_Z = jnp.sum(P_det_mu_Z_values)

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
    return P_dc + e_mis * (1 - jnp.exp(-eta_sys_values * mu_k)) + P_ap * D_mu_k / 2

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
    result = jnp.array(m_X_total) / jnp.array(n_X_values)
    return jnp.minimum(result, 0.5)  # Limit the error rate to a maximum of 0.5

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
    x = jnp.clip(x, 1e-10, 1 - 1e-10)  # Avoid log(0)
    return -x * jnp.log2(x) - (1 - x) * jnp.log2(1 - x)

def calculate_lambda_EC(n_X_values, f_EC, calculate_e_obs):
    """
    Calculates the error correction term.

    Parameters:
    - n_X_value (float): Total detected events in X basis.
    - f_EC (float): Error correction efficiency.
    - calculate_e_obs (float): Observed error rate.

    Returns:
    - float: Error correction term.
    """
    return n_X_values * f_EC * calculate_h(calculate_e_obs)

def calculate_sqrt_term(n_total, epsilon_sec):
    """
    Calculate the square root term used in uncertainty calculations for a given basis.

    Parameters:
    - n_total (float): Total number of events in the specified basis (X or Z).
    - epsilon_sec (float): Security parameter epsilon for secrecy.

    Returns:
    - float: Calculated square root term.
    """
    epsilon_sec = jnp.clip(epsilon_sec, 1e-10, None)  # Avoid log(0)
    return jnp.sqrt((n_total / 2) * jnp.log(21 / epsilon_sec))


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
    # Ensure inputs are JAX arrays
    mu_k_values = jnp.array(mu_k_values)
    p_mu_k_values = jnp.array(p_mu_k_values)
    
    # Calculate tau values
    def single_tau_n(n):
        tau_values = p_mu_k_values * jnp.exp(-mu_k_values) * (mu_k_values ** n) / calculate_factorial(n)
        return jnp.sum(tau_values)
    # Use vmap to apply over all `n_values`
    vmap_tau_n = vmap(single_tau_n)
    return vmap_tau_n(n)

def calculate_n_pm(mu_k_values, p_mu_k_values, n_mu_k, calculate_sqrt_term):
    """
    Calculate the bounds for a specific intensity level.

    Parameters:
    - mu_k_values (array): Intensity levels.
    - p_mu_k_values (array): Probabilities for intensity levels.
    - n_mu_k (float): Detected events for intensity.
    - calculate_sqrt_term (float): Uncertainty term.

    Returns:
    - tuple: (n_plus, n_minus).
    """
    n_plus = ((jnp.exp(mu_k_values)) / p_mu_k_values) * (n_mu_k + calculate_sqrt_term)
    n_minus = ((jnp.exp(mu_k_values)) / p_mu_k_values) * jnp.maximum(n_mu_k - calculate_sqrt_term, 0)  # Ensure non-negative lower bound
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

    return jnp.maximum(result, 0)

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

    return jnp.maximum(result, 0)

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
    m_mu_k_values = [e_mu_k * N_value * (P)**2 * p_mu_k for e_mu_k, p_mu_k in zip(e_mu_k_values, p_mu_k_values)]

    return m_mu_k_values

def calculate_m_pm(mu_k_values, p_mu_k_values, m_k, calculate_sqrt_term):
    """
    Calculates bounds on m_k for uncertainty.

    Parameters:
    - mu_k_values (array): Mean photon numbers.
    - p_mu_k_values (array): Probabilities for each intensity.
    - m_k (float): Error rate for intensity.
    - calculate_sqrt_term (float): Uncertainty term.

    Returns:
    - tuple: (m_plus, m_minus).
    """
    m_plus = ((jnp.exp(mu_k_values)) / p_mu_k_values) * (m_k + calculate_sqrt_term)
    m_minus = ((jnp.exp(mu_k_values)) / p_mu_k_values) * jnp.maximum(m_k - calculate_sqrt_term, 0)
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
    def true_fn(_):
        return 0.0

    def false_fn(_):
        term1 = (c + d) * (1 - b) * b / (c * d * jnp.log(2))
        term2 = jnp.log2((c + d) / (c * d * (1 - b) * b) * (21**2 / a**2))
        return jnp.sum(jnp.sqrt(term1 * term2))

    # Use `lax.cond` to handle the condition
    return lax.cond((b == 0) | (b == 1), true_fn, false_fn, operand=None)

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
    return jnp.minimum(result, 0.5)

def calculate_LastTwoTerm(epsilon_sec, epsilon_cor):
    """
    Calculates the final adjustment term in the security bound, dependent on the secrecy and correctness parameters.

    Parameters:
    - epsilon_sec (float): Secrecy parameter epsilon.
    - epsilon_cor (float): Correctness parameter epsilon.

    Returns:
    - float: Adjustment term for the key rate calculation.
    """
    return -6 * jnp.log2(21/epsilon_sec) - jnp.log2(2/epsilon_cor)

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
    l_value = S_X_0_values + S_X_1_values - S_X_1_values * binary_entropy_Phi_values - lambda_EC_values + calculate_LastTwoTerm(epsilon_sec, epsilon_cor)
    
    return l_value    

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


def validate_parameters_and_conditions():
    """
    Validates key conditions and parameter ranges for decoy-state QKD setup.

    Returns:
    - bool: True if all conditions and range checks are satisfied, else False.
    """
    # Core conditions for parameter relationships
    conditions = {
        "Condition 1: mu_1 > mu_2 + mu_3": mu_1 > mu_2 + mu_3,
        "Condition 2: mu_2 > mu_3 >= 0": mu_2 > mu_3 >= 0,
        "Condition 3: P_mu values sum to 1": abs(P_mu_1 + P_mu_2 + P_mu_3 - 1) < 1e-5,
        "Condition 4: P_X + P_Z sum to 1": abs(P_X_value + P_Z_value - 1) < 1e-5
    }
    
    # Print and evaluate each condition
    all_conditions_passed = True
    for condition, is_satisfied in conditions.items():
        status = "Pass" if is_satisfied else "Fail"
        print(f"{condition}: {status}")
        all_conditions_passed &= is_satisfied  # Update the final status

    # Bound checks for parameters
    bounds = {
        "Minimum Fiber Length": (min(Ls), (0.1, 200)),
        "Maximum Fiber Length": (max(Ls), (0.1, 200)),
        "Attenuation Coefficient (alpha)": (alpha, (0.1, 1)),
        "Dark Count Probability (P_dc)": (P_dc_value, (1e-8, 1e-5)),
        "After pulse Probability (P_ap)": (P_ap, (0, 0.1)),
        "System Transmittance (eta_sys)": (eta_sys_values, (1e-6, 1)),
        "Channel Transmittance (eta_ch)": (eta_ch_values, (1e-6, 1)),
        "Detector Efficiency (eta_Bob)": (eta_Bob, (0, 1)),
        "Secrecy Parameter (epsilon_sec)": (epsilon_sec, (1e-10, 1)),
        "Correctness Parameter (epsilon_cor)": (epsilon_cor, (1e-15, 1)),
        "Error Correction Efficiency (f_EC)": (f_EC, (1, 2)),
        "Detected Events in X Basis (n_X_values)": (n_X_values, (1e9, 1e11)),
        "Detected Events in Z Basis (n_Z)": (n_Z_mu_values, (1e8, 1e11)),  # List validation
        "Total Events in Z Basis (n_Z_total)": (n_Z_total, (1e9, 1e11)),  # Scalar validation
    }

    # Check each parameter's bounds and store out-of-bound values
    all_bounds_passed = True
    out_of_bound_params = []
    for name, (value, range_) in bounds.items():
        if isinstance(value, (list, np.ndarray, jnp.ndarray)):  # Handle lists/arrays
            out_of_bounds = [
                (i, v) for i, v in enumerate(value) if not (range_[0] <= v <= range_[1])
            ]
            if out_of_bounds:
                print(f"{name} out of bounds:")
                for idx, v in out_of_bounds:
                    print(f"  - Element {idx}: {v} (Expected range: {range_})")
                out_of_bound_params.append((name, out_of_bounds))
                all_bounds_passed = False
            else:
                print(f"{name}: All elements within bounds.")
        elif isinstance(value, (jnp.ndarray, np.ndarray)):  # Scalar-like array
            within_bounds = jnp.logical_and(range_[0] <= value, value <= range_[1]).all()
            status = "within bounds" if within_bounds else "out of bounds"
            print(f"{name}: {value} ({status}) - Expected range: {range_}")
            if not within_bounds:
                out_of_bound_params.append((name, value, range_))
                all_bounds_passed = False
        else:  # Handle single scalar values
            within_bounds = range_[0] <= value <= range_[1]
            status = "within bounds" if within_bounds else "out of bounds"
            print(f"{name}: {value} ({status}) - Expected range: {range_}")
            if not within_bounds:
                out_of_bound_params.append((name, value, range_))
                all_bounds_passed = False

    # Print out-of-bound values, if any
    if out_of_bound_params:
        print("\nThe following parameters are out of bounds:")
        for param in out_of_bound_params:
            if isinstance(param[1], list):  # List-like parameter
                name, out_of_bounds = param
                for idx, value in out_of_bounds:
                    print(f"  - {name} (Element {idx}): {value} (Expected range: {bounds[name][1]})")
            else:  # Scalar parameter
                name, value, range_ = param
                print(f"  - {name}: {value} (Expected range: {range_})")

    # Final validation status
    if all_conditions_passed and all_bounds_passed:
        print("\n✅ All conditions and parameter ranges are within expected bounds.")
        return True
    else:
        print("\n❌ One or more conditions/parameters are out of bounds.")
        return False
    
# Define the objective function
@jit
# Define the `objective` function with `alpha` and other parameters as arguments
def calculate_key_rates_and_metrics(params, L_values, n_X, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event): 
#  mu_k_values, eta_ch_values, p_mu_k_values, p_1 = mu_1, p_2 = mu_2, p_3 = P_mu_1 = 0.65, p_4 = P_mu_2 = 0.3, p_5 =        P_X_value = 5e-3
    mu_1, mu_2, P_mu_1, P_mu_2, P_X_value = params 
    mu_3 = 2e-4
    mu_k_values = jnp.array([mu_1, mu_2, mu_3])
    P_mu_3 = 1 - P_mu_1 - P_mu_2
    p_mu_k_values = jnp.array([P_mu_1, P_mu_2, P_mu_3])
       
    P_Z_value = 1 - P_X_value
    # n_X = jnp.array([10**s for s in range(6, 11)])  # Detected events in X basis: 10^6 to 10^10
    """Objective function to optimize key rate."""
# 1. Channel and system efficiencies
    eta_ch_values = calculate_eta_ch(L_values, alpha)  # Channel transmittance
    eta_sys_values = calculate_eta_sys(eta_Bob, eta_ch_values)  # System transmittance

    # 2. Detection probabilities for each intensity level
    D_mu_k_values = jnp.array([calculate_D_mu_k(mu_k, eta_sys_values, P_dc_value) for mu_k in mu_k_values])
    # 3. Error rates for each intensity level
    e_mu_k_values = jnp.array([calculate_e_mu_k(P_dc_value, e_mis, P_ap, D_mu_k, eta_sys_values, mu_k)
                    for D_mu_k, mu_k in zip(D_mu_k_values, mu_k_values)])
    # 4. Detection probabilities and events in the X basis
    sum_P_det_mu_X, P_det_mu_1, P_det_mu_2, P_det_mu_3, n_X_total, n_X_mu_1, n_X_mu_2, n_X_mu_3 = calculate_n_X_total(n_event, mu_1, mu_2, mu_3, P_mu_1, P_mu_2, P_mu_3, P_dc_value, eta_sys_values, P_X_value, n_X)
    sqrt_term_n_X = calculate_sqrt_term(n_X, epsilon_sec)  # Uncertainty in X basis

    # Organize detection probabilities and detected events
    n_X_mu_k_values = jnp.array([n_X_mu_1, n_X_mu_2, n_X_mu_3])

    #n_X_total = sum(n_X_mu_k_values)  # Total errors in X basis
    n_X_total = jnp.sum(n_X_mu_k_values)

    P_det_mu_values = [P_det_mu_1, P_det_mu_2, P_det_mu_3]

    n_plus_X_mu_1, n_minus_X_mu_1 = calculate_n_pm(mu_1, P_mu_1, n_X_mu_1, sqrt_term_n_X) # m_plus and m_minus for m_X_mu_1

    n_plus_X_mu_2, n_minus_X_mu_2 = calculate_n_pm(mu_2, P_mu_2, n_X_mu_2, sqrt_term_n_X) # m_plus and m_minus for m_X_mu_2

    n_plus_X_mu_3, n_minus_X_mu_3 = calculate_n_pm(mu_3, P_mu_3, n_X_mu_3, sqrt_term_n_X)

    # 5. Total pulses and events in Z basis
    N_values = calculate_N(n_X_total, p_mu_k_values, D_mu_k_values, P_X_value)
    sum_P_det_mu_Z, n_Z_total, n_Z_mu_1, n_Z_mu_2, n_Z_mu_3 = calculate_n_Z_total(N_values, p_mu_k_values, D_mu_k_values, P_Z_value, P_det_mu_values)

    sqrt_term_n_Z = calculate_sqrt_term(n_Z_total, epsilon_sec)  # Uncertainty in Z basis
    
    # Organize detected events in the Z basis
    n_Z_mu_values = jnp.array([n_Z_mu_1, n_Z_mu_2, n_Z_mu_3])
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
    m_X_mu_values = jnp.array([m_X_mu_1, m_X_mu_2, m_X_mu_3])

    # m_X_total = sum(m_X_mu_values)  # Total errors in X basis
    m_X_total = jnp.sum(m_X_mu_values) 
    
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

    m_Z_mu_values = jnp.array([n_X_mu_1, n_X_mu_2, n_X_mu_3])
    m_Z_total = jnp.sum(m_Z_mu_values)  # Total errors in Z basi

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

    penalty_mu1_sum = jnp.where(mu_1 > mu_2 + mu_3, 0.0, 1e6)
    penalty_mu2_ratio = jnp.where(mu_2 / mu_1 < 1, 0.0, 1e6)
    # This penalty works well for enforcing the sum of probabilities, but the tolerance (1e-10) might be too strict for numerical optimizations, leading to unnecessary penalties.
    penalty_sum = jnp.where(jnp.abs(P_mu_1 + P_mu_2 + P_mu_3 - 1) < 1e-6, 0.0, 1e6)
    penalty_P_mu_3 = jnp.where(P_mu_3 > 0, 0.0, 1e6)
    penalty_mu2_mu3 = jnp.where(mu_2 > mu_3, 0.0, 1e6)
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
    mu_3 = 2e-4  # Ensure mu_3 is defined
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

def objective_with_logging(params, *args):
    try:
        result = objective(params, *args)
        print(f"Params: {params}, Key Rate: {result}")
        return result
    except ZeroDivisionError:
        print(f"Division by zero detected with Params: {params}")
        return float('inf')  # Return a large penalty value
    except Exception as e:
        print(f"Error during computation with Params: {params}: {str(e)}")
        return float('inf')
    
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