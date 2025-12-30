import numpy as np
try:
    import jax.numpy as jnp
except ImportError:
    import numpy as jnp

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

def other_parameters():
    # other parameters
    alpha = 0.2  # Attenuation coefficient (dB/km), given in the paper
    eta_Bob = 0.1  # Detector efficiency, given in the paper
    P_ap = 0  # After-pulse probability
    f_EC = 1.16  # Error correction efficiency, given in the paper
    # secutity error 
    epsilon_sec = 1e-10 # is equal to kappa * secrecy length Kl, range around 1e-10 Scalar, as it is a single value throughout the calculations.
    # correlation error
    epsilon_cor = 1e-15 # given in the paper, discussed with range from 0 to 10e-15
    # Dark count probability
    n_event = 1  # for single photon event
    # Misalignment error probability
    kappa = 1e-15           # given in the paper
    return

def validate_parameters_and_conditions(mu_1, mu_2, mu_3, P_mu_1, P_mu_2, P_mu_3, P_X_value, P_Z_value, 
                                       Ls, alpha, P_dc_value, eta_sys_values, eta_ch_values, eta_Bob, 
                                       epsilon_sec, epsilon_cor, f_EC, n_X_values, n_Z_mu_values, n_Z_total):
    """
    Validates key conditions and parameter ranges for decoy-state QKD setup.

    Returns:
    - bool: True if all conditions and range checks are satisfied, else False.
    """
    # Core conditions for parameter relationships
    conditions = {
        "Condition 1: mu_1 > mu_2 + mu_3": mu_1 > mu_2 + mu_3,
        "Condition 2: mu_2 > mu_3 >= 0": mu_2 > mu_3 >= 0,
        "Condition 3: P_mu values sum to 1": abs(P_mu_1 + P_mu_2 + P_mu_3 - 1) < 1e-12,
        "Condition 4: P_X + P_Z sum to 1": abs(P_X_value + P_Z_value - 1) < 1e-12
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
