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

@jit
def calculate_key_rates_and_metrics(params, L_values, n_X, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event): 
    # Ensure params is 1D (5,) to match scalar unpacking
    params = jnp.ravel(params)
    mu_1, mu_2, P_mu_1, P_mu_2, P_X_value = params 
    mu_3 = 2e-4
    mu_k_values = jnp.array([mu_1, mu_2, mu_3])
    P_mu_3 = 1 - P_mu_1 - P_mu_2
    p_mu_k_values = jnp.array([P_mu_1, P_mu_2, P_mu_3])
    P_Z_value = 1 - P_X_value

    # 1. Channel and system efficiencies
    eta_ch_values = calculate_eta_ch(L_values, alpha)
    eta_sys_values = calculate_eta_sys(eta_Bob, eta_ch_values)

    # 2. Detection probabilities
    D_mu_k_values = jnp.array([calculate_D_mu_k(mu_k, eta_sys_values, P_dc_value) for mu_k in mu_k_values])
    
    # 3. Error rates
    e_mu_k_values = jnp.array([calculate_e_mu_k(P_dc_value, e_mis, P_ap, D_mu_k, eta_sys_values, mu_k)
                    for D_mu_k, mu_k in zip(D_mu_k_values, mu_k_values)])
    
    # 4. Detection probabilities and events (X basis)
    sum_P_det_mu_X, P_det_mu_1, P_det_mu_2, P_det_mu_3, n_X_total, n_X_mu_1, n_X_mu_2, n_X_mu_3 = calculate_n_X_total(n_event, mu_1, mu_2, mu_3, P_mu_1, P_mu_2, P_mu_3, P_dc_value, eta_sys_values, P_X_value, n_X)
    sqrt_term_n_X = calculate_sqrt_term(n_X, epsilon_sec)

    n_X_mu_k_values = jnp.array([n_X_mu_1, n_X_mu_2, n_X_mu_3])
    n_X_total = jnp.sum(n_X_mu_k_values)

    n_plus_X_mu_1, n_minus_X_mu_1 = calculate_n_pm(mu_1, P_mu_1, n_X_mu_1, sqrt_term_n_X)
    n_plus_X_mu_2, n_minus_X_mu_2 = calculate_n_pm(mu_2, P_mu_2, n_X_mu_2, sqrt_term_n_X)
    n_plus_X_mu_3, n_minus_X_mu_3 = calculate_n_pm(mu_3, P_mu_3, n_X_mu_3, sqrt_term_n_X)

    # 5. Total pulses and events (Z basis)
    N_values = calculate_N(n_X_total, p_mu_k_values, D_mu_k_values, P_X_value)
    
    # We need P_det_mu_values for Z total calculation
    P_det_mu_values = [P_det_mu_1, P_det_mu_2, P_det_mu_3]
    sum_P_det_mu_Z, n_Z_total, n_Z_mu_1, n_Z_mu_2, n_Z_mu_3 = calculate_n_Z_total(N_values, p_mu_k_values, D_mu_k_values, P_Z_value, P_det_mu_values)

    sqrt_term_n_Z = calculate_sqrt_term(n_Z_total, epsilon_sec)
    
    n_plus_Z_mu_1, n_minus_Z_mu_1 = calculate_n_pm(mu_1, P_mu_1, n_Z_mu_1, sqrt_term_n_Z)
    n_plus_Z_mu_2, n_minus_Z_mu_2 = calculate_n_pm(mu_2, P_mu_2, n_Z_mu_2, sqrt_term_n_Z)
    n_plus_Z_mu_3, n_minus_Z_mu_3 = calculate_n_pm(mu_3, P_mu_3, n_Z_mu_3, sqrt_term_n_Z)

    # 7. Security-related terms
    tau_0_values = calculate_tau_n(0, mu_k_values, p_mu_k_values)
    tau_1_values = calculate_tau_n(1, mu_k_values, p_mu_k_values)

    # 8. Error terms (X basis)
    m_X_mu_values = calculate_m_mu_k(e_mu_k_values, p_mu_k_values, N_values, P_X_value)
    m_X_mu_1, m_X_mu_2, m_X_mu_3 = m_X_mu_values[0], m_X_mu_values[1], m_X_mu_values[2]
    
    m_X_total = jnp.sum(jnp.array(m_X_mu_values))
    sqrt_term_m_X = calculate_sqrt_term(m_X_total, epsilon_sec)
    
    m_plus_X_mu_1, m_minus_X_mu_1 = calculate_m_pm(mu_1, P_mu_1, m_X_mu_1, sqrt_term_m_X)
    m_plus_X_mu_2, m_minus_X_mu_2 = calculate_m_pm(mu_2, P_mu_2, m_X_mu_2, sqrt_term_m_X)
    m_plus_X_mu_3, m_minus_X_mu_3 = calculate_m_pm(mu_3, P_mu_3, m_X_mu_3, sqrt_term_m_X)

    e_obs_X_values = calculate_e_obs(m_X_total, n_X)

    # 9. Error terms (Z basis)
    m_Z_mu_values = calculate_m_mu_k(e_mu_k_values, p_mu_k_values, N_values, P_Z_value)
    m_Z_mu_1, m_Z_mu_2, m_Z_mu_3 = m_Z_mu_values[0], m_Z_mu_values[1], m_Z_mu_values[2]

    # Replicating original behavior (sum of n_X used for m_Z total in original code? 
    # Standard logic usually sums m_Z_mu. Keeping your logic: sum of n_X tensor)
    # Warning: Original code summed n_X values for m_Z total. Assuming this is intentional logic.
    m_Z_total = jnp.sum(jnp.array([n_X_mu_1, n_X_mu_2, n_X_mu_3])) 

    sqrt_term_m_Z = calculate_sqrt_term(m_Z_total, epsilon_sec)
    
    m_plus_Z_mu_1, m_minus_Z_mu_1 = calculate_m_pm(mu_1, P_mu_1, m_Z_mu_1, sqrt_term_m_Z)
    m_plus_Z_mu_2, m_minus_Z_mu_2 = calculate_m_pm(mu_2, P_mu_2, m_Z_mu_2, sqrt_term_m_Z)
    m_plus_Z_mu_3, m_minus_Z_mu_3 = calculate_m_pm(mu_3, P_mu_3, m_Z_mu_3, sqrt_term_m_Z)

    # 10. Contributions for single-photon events
    S_X_0_values = calculate_S_0(tau_0_values, mu_2, mu_3, n_minus_X_mu_3, n_plus_X_mu_2)
    S_Z_0_values = calculate_S_0(tau_0_values, mu_2, mu_3, n_minus_Z_mu_3, n_plus_Z_mu_2)

    S_X_1_values = calculate_S_1(tau_1_values, mu_1, mu_2, mu_3,n_minus_X_mu_2, n_plus_X_mu_3, n_plus_X_mu_1, S_X_0_values, tau_0_values)
    S_Z_1_values = calculate_S_1(tau_1_values, mu_1, mu_2, mu_3,n_minus_Z_mu_2, n_plus_Z_mu_3, n_plus_Z_mu_1, S_Z_0_values, tau_0_values)

    # 11. Security bounds
    v_Z_1_values = calculate_v_1(tau_1_values, m_plus_Z_mu_2, m_minus_Z_mu_3, mu_2, mu_3)

    gamma_results = calculate_gamma(epsilon_sec, v_Z_1_values / (S_Z_1_values + 1e-12), S_Z_1_values, S_X_1_values)
    Phi_X_values = calculate_Phi(v_Z_1_values, S_Z_1_values, gamma_results)
    binary_entropy_Phi_values = calculate_h(Phi_X_values)

    # 12. Final key rate
    lambda_EC_values = calculate_lambda_EC(n_X, f_EC, e_obs_X_values)
    l_calculated_values = calculate_l(S_X_0_values, S_X_1_values, binary_entropy_Phi_values,
                                        lambda_EC_values, epsilon_sec, epsilon_cor)
    key_rates = calculate_R(l_calculated_values, N_values)

    return (key_rates, eta_ch_values, S_X_0_values, S_Z_0_values, S_X_1_values, S_Z_1_values, tau_0_values, tau_1_values, e_mu_k_values, e_obs_X_values, v_Z_1_values, gamma_results, Phi_X_values, binary_entropy_Phi_values, lambda_EC_values, l_calculated_values)

def penalty(key_rates, mu_1, mu_2, mu_3, P_mu_1, P_mu_2, P_mu_3):
    """Penalty function to enforce constraints."""
    # Compute penalties with JAX operations
    
    # mu_1 > mu_2 + mu_3
    penalty_mu1_sum = jnp.where(mu_1 > mu_2 + mu_3, 0.0, 1e250)
    
    # mu_2 / mu_1 < 1 (mu_2 < mu_1)
    penalty_mu2_ratio = jnp.where(mu_2 / mu_1 < 1, 0.0, 1e250)
    
    # Sum of probabilities ~ 1
    penalty_sum = jnp.where(jnp.abs(P_mu_1 + P_mu_2 + P_mu_3 - 1) < 1e-12, 0.0, 1e250)
    
    # P_mu_3 > 0
    penalty_P_mu_3 = jnp.where(P_mu_3 > 0, 0.0, 1e250)
    
    # mu_2 > mu_3
    penalty_mu2_mu3 = jnp.where(mu_2 > mu_3, 0.0, 1e250)
    
    total_penalty = penalty_mu1_sum + penalty_mu2_ratio + penalty_sum + penalty_mu2_mu3 + penalty_P_mu_3
    
    # Apply penalty to key rates (Key Rate - Penalty)
    # Since we maximize Key Rate, we want to subtract positive penalty.
    penalized_key_rates = key_rates - total_penalty
    return penalized_key_rates

def objective(params, L_values, n_X, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event):
    """
    Objective function with penalty applied to key rates.
    """
    # Unpack parameters
    mu_1, mu_2, P_mu_1, P_mu_2, P_X_value = params
    mu_3 = 2e-4  
    P_mu_3 = 1 - P_mu_1 - P_mu_2 

    # Compute metrics
    metrics = calculate_key_rates_and_metrics(params, L_values, n_X, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event)
    key_rates = metrics[0]

    # Apply penalty
    penalized_key_rates = penalty(key_rates, mu_1, mu_2, mu_3, P_mu_1, P_mu_2, P_mu_3)

    # Return all metrics with penalized key rates as first element
    return (penalized_key_rates,) + metrics[1:]

def scalar_objective(params, L_values, n_X, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event):
    """
    Returns the negative Key Rate (scalar) for minimization.
    """
    # Helper to unpack and calculate penalty
    params = jnp.ravel(params)
    mu_1, mu_2, P_mu_1, P_mu_2, P_X_value = params
    mu_3 = 2e-4
    P_mu_3 = 1 - P_mu_1 - P_mu_2
    
    metrics = calculate_key_rates_and_metrics(params, L_values, n_X, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event)
    key_rates = metrics[0]
    
    penalized_key_rate = penalty(key_rates, mu_1, mu_2, mu_3, P_mu_1, P_mu_2, P_mu_3)
    
    # Minimize negative key rate (Maximize positive key rate)
    return -jnp.sum(penalized_key_rate)

jit_scalar_objective = jit(scalar_objective)
objective_val_and_grad = jit(jax.value_and_grad(scalar_objective, argnums=0))
