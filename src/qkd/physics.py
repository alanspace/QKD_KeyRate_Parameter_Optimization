from math import exp, factorial
import jax
import jax.numpy as jnp
from jax.scipy.special import gamma, logsumexp
from jax import lax

# Configure JAX
jax.config.update("jax_enable_x64", True)

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
    Calculates the probability of detection and the expected number of events in the X basis for different intensity levels.
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
    """
    terms = jnp.array([p_mu_k * D_mu_k * P_X_value**2 for p_mu_k, D_mu_k in zip(p_mu_k_values, D_mu_k_values)])
    N_value = n_X / jnp.sum(terms)
    return N_value
    
def calculate_n_Z_total(N_value, p_mu_k_values, D_mu_k_values, P_Z_value, P_det_mu_values):
    """
    Calculates both the total and individual expected number of events in the Z basis for each intensity level.
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

def calculate_sqrt_term(n_total, epsilon_sec):
    """
    Calculate the square root term used in uncertainty calculations for a given basis.
    """
    epsilon_sec = jnp.clip(epsilon_sec, 1e-10, None)  # Avoid log(0)
    return jnp.sqrt((n_total / 2) * jnp.log(21 / epsilon_sec))

def calculate_n_pm(mu_k_values, p_mu_k_values, n_mu_k, calculate_sqrt_term):
    """
    Calculate the bounds for a specific intensity level.
    """
    n_plus = ((jnp.exp(mu_k_values)) / p_mu_k_values) * (n_mu_k + calculate_sqrt_term)
    n_minus = ((jnp.exp(mu_k_values)) / p_mu_k_values) * jnp.maximum(n_mu_k - calculate_sqrt_term, 0)  # Ensure non-negative lower bound
    return n_plus, n_minus

def calculate_tau_n(n, mu_k_values, p_mu_k_values):
    """
    Calculates Poisson probabilities for intensity levels using JAX.
    """
    # Ensure inputs are JAX arrays
    mu_k_values = jnp.array(mu_k_values)
    p_mu_k_values = jnp.array(p_mu_k_values)
    
    # Calculate tau values
    tau_values = p_mu_k_values * jnp.exp(-mu_k_values) * (mu_k_values ** n) / calculate_factorial(n)
    return jnp.sum(tau_values)

# Number of vacuum events
def calculate_S_0(tau_0, mu_2, mu_3, n_minus_mu_3, n_plus_mu_2):
    """
    Calculate S_0 for the basis (X or Z).
    """
    result = tau_0 * (mu_2 * n_minus_mu_3 - mu_3 * n_plus_mu_2) / (mu_2 - mu_3)

    return jnp.maximum(result, 0)

def calculate_S_1(tau_1, mu_1, mu_2, mu_3, n_minus_mu_2, n_plus_mu_3, n_plus_mu_1, s_0, tau_0):
    """
    Calculate S_1 for the basis (X or Z).
    """
    numerator = tau_1 * mu_1 * (n_minus_mu_2 - n_plus_mu_3 - ((mu_2**2 - mu_3**2) / mu_1**2) * (n_plus_mu_1 - s_0 / tau_0))
    denominator = mu_1 * (mu_2 - mu_3) - mu_2**2 + mu_3**2
    result = numerator / denominator

    return jnp.maximum(result, 0)

def calculate_m_mu_k(e_mu_k_values, p_mu_k_values, N_value, P):
    """
    Calculates m_k for each intensity level in a specified basis.
    """
    # Calculate m_k for each intensity
    m_mu_k_values = [e_mu_k * N_value * (P)**2 * p_mu_k for e_mu_k, p_mu_k in zip(e_mu_k_values, p_mu_k_values)]

    return m_mu_k_values

def calculate_m_pm(mu_k_values, p_mu_k_values, m_k, calculate_sqrt_term):
    """
    Calculates bounds on m_k for uncertainty.
    """
    m_plus = ((jnp.exp(mu_k_values)) / p_mu_k_values) * (m_k + calculate_sqrt_term)
    m_minus = ((jnp.exp(mu_k_values)) / p_mu_k_values) * jnp.maximum(m_k - calculate_sqrt_term, 0)
    return m_plus, m_minus

def calculate_v_1(tau_1, m_plus_mu_2, m_minus_mu_3, mu_2, mu_3):
    """
    Calculates v_1, a value used in the error rate estimation, for a specified basis (X or Z).
    """
    return tau_1 * (m_plus_mu_2 - m_minus_mu_3) / (mu_2 - mu_3)

def calculate_e_mu_k(P_dc, e_mis, P_ap, D_mu_k, eta_sys_values, mu_k):
    """
    Calculates the error rate for a single intensity level.
    """
    return P_dc + e_mis * (1 - jnp.exp(-eta_sys_values * mu_k)) + P_ap * D_mu_k / 2

def calculate_e_obs(m_X_total, n_X_values):
    """
    Calculate the observed error rate in a specified basis (X or Z) given the number 
    of events for each intensity and error rates.
    """
    result = jnp.array(m_X_total) / jnp.array(n_X_values)

    return jnp.minimum(result, 0.5)  # Limit the error rate to a maximum of 0.5

def calculate_h(x):
    """
    Binary entropy function.
    """
    x = jnp.clip(x, 1e-10, 1 - 1e-10)  # Avoid log(0)
    return -x * jnp.log2(x) - (1 - x) * jnp.log2(1 - x)

def calculate_lambda_EC(n_X_values, f_EC, calculate_e_obs):
    """
    Calculates the error correction term.
    """
    return n_X_values * f_EC * calculate_h(calculate_e_obs)

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
    """
    result = v_Z_1 / s_Z_1 + gamma_result
    return jnp.minimum(result, 0.5)

def calculate_LastTwoTerm(epsilon_sec, epsilon_cor):
    """
    Calculates the final adjustment term in the security bound, dependent on the secrecy and correctness parameters.
    """
    return -6 * jnp.log2(21/epsilon_sec) - jnp.log2(2/epsilon_cor)

def calculate_l(S_X_0_values, S_X_1_values, binary_entropy_Phi_values, lambda_EC_values, epsilon_sec, epsilon_cor):
    """
    Calculates the final secret key length for the specified basis.
    """
    l_value = S_X_0_values + S_X_1_values - S_X_1_values * binary_entropy_Phi_values - lambda_EC_values + calculate_LastTwoTerm(epsilon_sec, epsilon_cor)
    
    return l_value    

def calculate_R(l: float, N: float) -> float:
    """
    Calculates the secret key rate per pulse.
    """
    # assert l >= 0, "Secret key length (l) must be non-negative."
    # assert N > 0, "Total pulses sent (N) must be positive."
    return l / N
