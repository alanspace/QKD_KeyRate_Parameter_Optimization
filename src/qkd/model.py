import jax
import jax.numpy as jnp
from jax import jit

from .physics import (
    calculate_eta_ch, calculate_eta_sys, calculate_D_mu_k, calculate_e_mu_k,
    calculate_n_X_total, calculate_sqrt_term, calculate_n_pm, calculate_N,
    calculate_n_Z_total, calculate_tau_n, calculate_m_mu_k, calculate_m_pm,
    calculate_e_obs, calculate_S_0, calculate_S_1, calculate_v_1,
    calculate_gamma, calculate_Phi, calculate_h, calculate_lambda_EC,
    calculate_l, calculate_R
)

# Define the `objective` function with `alpha` and other parameters as arguments
@jit
def calculate_key_rates_and_metrics(params, L_values, n_X, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event): 
    mu_1, mu_2, P_mu_1, P_mu_2, P_X_value = params 
    mu_3 = 2e-4
    mu_k_values = jnp.array([mu_1, mu_2, mu_3])
    P_mu_3 = 1 - P_mu_1 - P_mu_2
    p_mu_k_values = jnp.array([P_mu_1, P_mu_2, P_mu_3])
       
    P_Z_value = 1 - P_X_value
    P_Z_value = 1 - P_X_value
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

    n_X_mu_k_values = jnp.array([n_X_mu_1, n_X_mu_2, n_X_mu_3])
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
    
    m_X_mu_values = jnp.array([m_X_mu_1, m_X_mu_2, m_X_mu_3])
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

    # Replicating original behavior
    m_Z_mu_values_tensor = jnp.array([n_X_mu_1, n_X_mu_2, n_X_mu_3]) 
    m_Z_total = jnp.sum(m_Z_mu_values_tensor)  # Total errors in Z basi

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

    gamma_results = calculate_gamma(epsilon_sec, v_Z_1_values / (S_Z_1_values + 1e-12), S_Z_1_values, S_X_1_values)
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

    penalty_mu1_sum = jnp.where(mu_1 > mu_2 + mu_3, 0.0, 1e250)
    penalty_mu2_ratio = jnp.where(mu_2 / mu_1 < 1, 0.0, 1e250)
    # This penalty works well for enforcing the sum of probabilities, but the tolerance (1e-10) might be too strict for numerical optimizations, leading to unnecessary penalties.
    penalty_sum = jnp.where(jnp.abs(P_mu_1 + P_mu_2 + P_mu_3 - 1) < 1e-12, 0.0, 1e250)
    penalty_P_mu_3 = jnp.where(P_mu_3 > 0, 0.0, 1e250)
    penalty_mu2_mu3 = jnp.where(mu_2 > mu_3, 0.0, 1e250)
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
        l_calculated_values
    )


def scalar_objective(params, L_values, n_X, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event):
    """
    Returns the negative Key Rate (scalar) for minimization.
    """
    metrics = calculate_key_rates_and_metrics(
        params, L_values, n_X, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event
    )
    key_rates = metrics[0]
    
    # Recalculate parameters for penalty
    mu_1, mu_2, P_mu_1, P_mu_2, P_X_value = params
    mu_3 = 2e-4
    P_mu_3 = 1 - P_mu_1 - P_mu_2
    
    penalized_key_rate = penalty(key_rates, mu_1, mu_2, mu_3, P_mu_1, P_mu_2, P_mu_3)
    
    # We want to MAXIMIZE key rate, so we MINIMIZE negative key rate
    # Returning scalar (assuming single L if optimizing point-wise, 
    # BUT current logic handles arrays. For optimization, we usually optimize for specific L)
    
    # Assuming input L is a scalar or we optimize 'mean' key rate over range (unlikely).
    # Usually dual_annealing calls this with specific context.
    # In the notebook: L is passed. If L is scalar, key_rate is scalar-ish array.
    
    return -jnp.sum(penalized_key_rate) # Sum handles both scalar and single-element array

# JIT compile the scalar objective
jit_scalar_objective = jit(scalar_objective)

# Create value_and_grad function
# This returns (loss, grads) tuple
objective_val_and_grad = jit(jax.value_and_grad(scalar_objective, argnums=0))
