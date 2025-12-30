import numpy as np
# No JIT needed for numpy

from .physics_numpy import (
    calculate_eta_ch, calculate_eta_sys, calculate_D_mu_k, calculate_e_mu_k,
    calculate_n_X_total, calculate_sqrt_term, calculate_n_pm, calculate_N,
    calculate_n_Z_total, calculate_tau_n, calculate_m_mu_k, calculate_m_pm,
    calculate_e_obs, calculate_S_0, calculate_S_1, calculate_v_1,
    calculate_gamma, calculate_Phi, calculate_h, calculate_lambda_EC,
    calculate_l, calculate_R
)

def calculate_key_rates_and_metrics(params, L_values, n_X, alpha, eta_Bob, P_dc_value, epsilon_sec, epsilon_cor, f_EC, e_mis, P_ap, n_event): 
    # Unpack
    mu_1, mu_2, P_mu_1, P_mu_2, P_X_value = params 
    mu_3 = 2e-4
    mu_k_values = np.array([mu_1, mu_2, mu_3])
    P_mu_3 = 1 - P_mu_1 - P_mu_2
    p_mu_k_values = np.array([P_mu_1, P_mu_2, P_mu_3])
       
    P_Z_value = 1 - P_X_value

    """Objective function to optimize key rate."""
# 1. Channel and system efficiencies
    eta_ch_values = calculate_eta_ch(L_values, alpha)  # Channel transmittance
    eta_sys_values = calculate_eta_sys(eta_Bob, eta_ch_values)  # System transmittance

    # 2. Detection probabilities for each intensity level
    D_mu_k_values = np.array([calculate_D_mu_k(mu_k, eta_sys_values, P_dc_value) for mu_k in mu_k_values])
    # 3. Error rates for each intensity level
    e_mu_k_values = np.array([calculate_e_mu_k(P_dc_value, e_mis, P_ap, D_mu_k, eta_sys_values, mu_k)
                    for D_mu_k, mu_k in zip(D_mu_k_values, mu_k_values)])
    # 4. Detection probabilities and events in the X basis
    sum_P_det_mu_X, P_det_mu_1, P_det_mu_2, P_det_mu_3, n_X_total, n_X_mu_1, n_X_mu_2, n_X_mu_3 = calculate_n_X_total(n_event, mu_1, mu_2, mu_3, P_mu_1, P_mu_2, P_mu_3, P_dc_value, eta_sys_values, P_X_value, n_X)
    sqrt_term_n_X = calculate_sqrt_term(n_X, epsilon_sec)  # Uncertainty in X basis

    # Organize detection probabilities and detected events
    n_X_mu_k_values = np.array([n_X_mu_1, n_X_mu_2, n_X_mu_3])

    #n_X_total = sum(n_X_mu_k_values)  # Total errors in X basis
    n_X_total = np.sum(n_X_mu_k_values)

    P_det_mu_values = [P_det_mu_1, P_det_mu_2, P_det_mu_3]

    n_plus_X_mu_1, n_minus_X_mu_1 = calculate_n_pm(mu_1, P_mu_1, n_X_mu_1, sqrt_term_n_X) # m_plus and m_minus for m_X_mu_1

    n_plus_X_mu_2, n_minus_X_mu_2 = calculate_n_pm(mu_2, P_mu_2, n_X_mu_2, sqrt_term_n_X) # m_plus and m_minus for m_X_mu_2

    n_plus_X_mu_3, n_minus_X_mu_3 = calculate_n_pm(mu_3, P_mu_3, n_X_mu_3, sqrt_term_n_X)

    # 5. Total pulses and events in Z basis
    N_values = calculate_N(n_X_total, p_mu_k_values, D_mu_k_values, P_X_value)
    sum_P_det_mu_Z, n_Z_total, n_Z_mu_1, n_Z_mu_2, n_Z_mu_3 = calculate_n_Z_total(N_values, p_mu_k_values, D_mu_k_values, P_Z_value, P_det_mu_values)

    sqrt_term_n_Z = calculate_sqrt_term(n_Z_total, epsilon_sec)  # Uncertainty in Z basis
    
    # Organize detected events in the Z basis
    n_Z_mu_values = np.array([n_Z_mu_1, n_Z_mu_2, n_Z_mu_3])
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
    m_X_mu_values = np.array([m_X_mu_1, m_X_mu_2, m_X_mu_3])

    # m_X_total = sum(m_X_mu_values)  # Total errors in X basis
    m_X_total = np.sum(m_X_mu_values) 
    
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

    m_Z_mu_values_tensor = np.array([n_X_mu_1, n_X_mu_2, n_X_mu_3]) 
    m_Z_total = np.sum(m_Z_mu_values_tensor)  # Total errors in Z basi

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

    return [key_rates] # Return as list to match original signature output[0]
