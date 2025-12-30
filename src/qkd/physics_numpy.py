from math import exp, factorial
import numpy as np
from scipy.special import gamma

def calculate_factorial(n):
    """
    Calculate the factorial using the gamma function.
    Factorial of n is gamma(n + 1).
    """
    return gamma(n + 1)

def calculate_eta_ch(L, alpha):
    eta = 10 ** (-alpha * L / 10)
    return eta 

def calculate_eta_sys(eta_Bob, eta_ch):
    return eta_Bob * eta_ch

def calculate_D_mu_k(mu_k, eta_sys_values, P_dc):
    return 1 - (1 - 2 * P_dc) * np.exp(-eta_sys_values * mu_k)

def calculate_n_X_total(n_event, mu_1, mu_2, mu_3, P_mu_1, P_mu_2, P_mu_3, P_dc, eta_sys_value, P_X_value, n_X_values):
     
    # Calculate the Poisson probabilities for each mu
    P_n_given_mu_1 = (mu_1**n_event * np.exp(-mu_1))/ calculate_factorial(n_event)
    P_n_given_mu_2 = (mu_2**n_event * np.exp(-mu_2)) / calculate_factorial(n_event)
    P_n_given_mu_3 = (mu_3**n_event * np.exp(-mu_3)) / calculate_factorial(n_event)
    
    # Calculate detection probabilities for each mu under channel condition
    D_mu_1 = 1 - np.exp(-mu_1* eta_sys_value) + 2 * P_dc * np.exp(-mu_1 * eta_sys_value)
    D_mu_2 = 1 - np.exp(-mu_2* eta_sys_value) + 2 * P_dc * np.exp(-mu_2 * eta_sys_value)
    D_mu_3 = 1 - np.exp(-mu_3* eta_sys_value) + 2 * P_dc * np.exp(-mu_3 * eta_sys_value)
    
    # Calculate joint detection probabilities
    P_det_mu_1 = D_mu_1 * P_mu_1
    P_det_mu_2 = D_mu_2 * P_mu_2
    P_det_mu_3 = D_mu_3 * P_mu_3
    
    # Calculate detection probabilities in the X basis
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
    terms = np.array([p_mu_k * D_mu_k * P_X_value**2 for p_mu_k, D_mu_k in zip(p_mu_k_values, D_mu_k_values)])
    N_value = n_X / np.sum(terms)
    return N_value
    
def calculate_n_Z_total(N_value, p_mu_k_values, D_mu_k_values, P_Z_value, P_det_mu_values):
    n_Z_mu_values = np.array([N_value * D_mu_k * p_mu_k * (P_Z_value)**2 for p_mu_k, D_mu_k in zip(p_mu_k_values, D_mu_k_values)])
    n_Z_total = np.sum(n_Z_mu_values)

    P_det_mu_Z_values = np.array([P_det_mu * (P_Z_value)**2 for P_det_mu in P_det_mu_values])
    sum_P_det_mu_Z = np.sum(P_det_mu_Z_values)

    n_Z_mu_1, n_Z_mu_2, n_Z_mu_3 = n_Z_mu_values

    return sum_P_det_mu_Z, n_Z_total, n_Z_mu_1, n_Z_mu_2, n_Z_mu_3

def calculate_sqrt_term(n_total, epsilon_sec):
    epsilon_sec = np.clip(epsilon_sec, 1e-10, None) 
    return np.sqrt((n_total / 2) * np.log(21 / epsilon_sec))

def calculate_n_pm(mu_k_values, p_mu_k_values, n_mu_k, calculate_sqrt_term):
    n_plus = ((np.exp(mu_k_values)) / p_mu_k_values) * (n_mu_k + calculate_sqrt_term)
    n_minus = ((np.exp(mu_k_values)) / p_mu_k_values) * np.maximum(n_mu_k - calculate_sqrt_term, 0)
    return n_plus, n_minus

def calculate_tau_n(n, mu_k_values, p_mu_k_values):
    mu_k_values = np.array(mu_k_values)
    p_mu_k_values = np.array(p_mu_k_values)
    tau_values = p_mu_k_values * np.exp(-mu_k_values) * (mu_k_values ** n) / calculate_factorial(n)
    return np.sum(tau_values)

def calculate_S_0(tau_0, mu_2, mu_3, n_minus_mu_3, n_plus_mu_2):
    result = tau_0 * (mu_2 * n_minus_mu_3 - mu_3 * n_plus_mu_2) / (mu_2 - mu_3)
    return np.maximum(result, 0)

def calculate_S_1(tau_1, mu_1, mu_2, mu_3, n_minus_mu_2, n_plus_mu_3, n_plus_mu_1, s_0, tau_0):
    numerator = tau_1 * mu_1 * (n_minus_mu_2 - n_plus_mu_3 - ((mu_2**2 - mu_3**2) / mu_1**2) * (n_plus_mu_1 - s_0 / tau_0))
    denominator = mu_1 * (mu_2 - mu_3) - mu_2**2 + mu_3**2
    result = numerator / denominator
    return np.maximum(result, 0)

def calculate_m_mu_k(e_mu_k_values, p_mu_k_values, N_value, P):
    m_mu_k_values = [e_mu_k * N_value * (P)**2 * p_mu_k for e_mu_k, p_mu_k in zip(e_mu_k_values, p_mu_k_values)]
    return m_mu_k_values

def calculate_m_pm(mu_k_values, p_mu_k_values, m_k, calculate_sqrt_term):
    m_plus = ((np.exp(mu_k_values)) / p_mu_k_values) * (m_k + calculate_sqrt_term)
    m_minus = ((np.exp(mu_k_values)) / p_mu_k_values) * np.maximum(m_k - calculate_sqrt_term, 0)
    return m_plus, m_minus

def calculate_v_1(tau_1, m_plus_mu_2, m_minus_mu_3, mu_2, mu_3):
    return tau_1 * (m_plus_mu_2 - m_minus_mu_3) / (mu_2 - mu_3)

def calculate_e_mu_k(P_dc, e_mis, P_ap, D_mu_k, eta_sys_values, mu_k):
    return P_dc + e_mis * (1 - np.exp(-eta_sys_values * mu_k)) + P_ap * D_mu_k / 2

def calculate_e_obs(m_X_total, n_X_values):
    result = np.array(m_X_total) / np.array(n_X_values)
    return np.minimum(result, 0.5)

def calculate_h(x):
    x = np.clip(x, 1e-10, 1 - 1e-10)
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)

def calculate_lambda_EC(n_X_values, f_EC, calculate_e_obs):
    return n_X_values * f_EC * calculate_h(calculate_e_obs)

def calculate_gamma(a, b, c, d):
    # This replaces lax.cond.
    # Since we are likely processing scalars in the web app, or simple arrays
    # But calculate_gamma in original physics.py handles a specific condition logic.
    # If b=0 or b=1, return 0.0. Else complicated logic.
    
    # We can implement a vectorized version using np.where
    
    # Ensure inputs are arrays
    a = np.atleast_1d(a)
    b = np.atleast_1d(b)
    c = np.atleast_1d(c)
    d = np.atleast_1d(d)
    
    term1 = (c + d) * (1 - b) * b / (c * d * np.log(2))
    # Avoid division by zero in log2 argument safety
    safe_arg = (c + d) / (c * d * (1 - b) * b) * (21**2 / a**2)
    safe_arg = np.maximum(safe_arg, 1e-20)
    term2 = np.log2(safe_arg)
    
    false_val = np.sum(np.sqrt(term1 * term2))
    
    # Since this function returns a single scalar sum in the original JAX code 
    # (jnp.sum(jnp.sqrt(term1 * term2))), we replicate that.
    # But wait, lax.cond checks (b==0) | (b==1). If b is an array, this check is ambiguous.
    # In JAX code: `lax.cond((b == 0) | (b == 1), ...)`
    # This implies b is likely treated as a scalar or all-or-nothing check.
    # Given the context in model.py:
    # gamma_results = calculate_gamma(epsilon_sec, v_Z_1_values / (S_Z_1_values + 1e-12), S_Z_1_values, S_X_1_values)
    # v_Z_1_values and S_Z_1_values seem to be derived from scalars in single-point evaluation.
    # But calculate_key_rates_and_metrics takes L_values (array?).
    # If L_values is scalar, then b is scalar.
    
    # Let's assume scalar for now, as that's what the web app passes mostly, 
    # or handle vectorized if needed.
    
    if np.any((b==0) | (b==1)):
         return 0.0
    
    return false_val

def calculate_Phi(v_Z_1, s_Z_1, gamma_result):
    result = v_Z_1 / s_Z_1 + gamma_result
    return np.minimum(result, 0.5)

def calculate_LastTwoTerm(epsilon_sec, epsilon_cor):
    return -6 * np.log2(21/epsilon_sec) - np.log2(2/epsilon_cor)

def calculate_l(S_X_0_values, S_X_1_values, binary_entropy_Phi_values, lambda_EC_values, epsilon_sec, epsilon_cor):
    l_value = S_X_0_values + S_X_1_values - S_X_1_values * binary_entropy_Phi_values - lambda_EC_values + calculate_LastTwoTerm(epsilon_sec, epsilon_cor)
    return l_value    

def calculate_R(l: float, N: float) -> float:
    return l / N
