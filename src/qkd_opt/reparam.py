"""Unconstrained reparameterisation theta in R^5 -> feasible params.

u = U_LO + (1 - 2 U_LO) * sigmoid(theta) in (0, 1)^5, then
    mu2 = mu3 + (MU2_MAX - mu3) u1                 -> mu3 < mu2 < MU2_MAX
    mu1 = (mu2 + mu3) + (MU1_MAX - mu2 - mu3) u0   -> mu2 + mu3 < mu1 < MU1_MAX
    P_mu1 = u2,  P_mu2 = (1 - u2) u3               -> P_mu1, P_mu2, P_mu3 = 1-P_mu1-P_mu2 all > 0
    q_x = u4                                       -> 0 < q_x < 1
so every constraint of the protocol (mu1 > mu2 + mu3, mu2 > mu3, probabilities in the open
simplex) holds by construction. The intensity caps MU1_MAX = 0.9, MU2_MAX = 0.5 are the box
used by all legacy variants (kept for comparability).

U_LO keeps u strictly inside (0, 1) in float64, so no probability can round to exactly 0.
A coordinate with u within ACTIVE_TOL of 0 or 1 is reported as an active bound.
"""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

MU1_MAX = 0.9
MU2_MAX = 0.5
U_LO = 1e-10
ACTIVE_TOL = 1e-6
PARAM_NAMES = ("mu_1", "mu_2", "P_mu_1", "P_mu_2", "P_X")
# what u_i -> 0 / u_i -> 1 means physically
BOUND_NAMES = (
    ("mu1->mu2+mu3", "mu1->MU1_MAX"),
    ("mu2->mu3", "mu2->MU2_MAX"),
    ("P_mu1->0", "P_mu1->1"),
    ("P_mu2->0", "P_mu3->0"),
    ("q_x->0", "q_x->1"),
)


def u_of_theta(theta):
    return U_LO + (1.0 - 2.0 * U_LO) * jax.nn.sigmoid(theta)


def params_of_u(u, mu3):
    u0, u1, u2, u3, u4 = u[..., 0], u[..., 1], u[..., 2], u[..., 3], u[..., 4]
    mu2 = mu3 + (MU2_MAX - mu3) * u1
    mu1 = (mu2 + mu3) + (MU1_MAX - mu2 - mu3) * u0
    P1 = u2
    P2 = (1.0 - u2) * u3
    return jnp.stack([mu1, mu2, P1, P2, u4], axis=-1)


def params_of_theta(theta, mu3):
    return params_of_u(u_of_theta(theta), mu3)


def u_of_params(p, mu3):
    """Inverse map (numpy, host side). Infeasible inputs are projected to the nearest interior u."""
    p = np.asarray(p, dtype=np.float64)
    mu1, mu2, P1, P2, qx = (p[..., i] for i in range(5))
    u1 = (mu2 - mu3) / (MU2_MAX - mu3)
    u0 = (mu1 - mu2 - mu3) / (MU1_MAX - mu2 - mu3)
    u2 = P1
    u3 = np.where(P1 < 1, P2 / np.maximum(1.0 - P1, 1e-300), 0.5)
    u = np.stack([u0, u1, u2, u3, qx], axis=-1)
    return np.clip(u, U_LO * 2, 1.0 - U_LO * 2)


def theta_of_u(u):
    u = np.asarray(u, dtype=np.float64)
    s = (u - U_LO) / (1.0 - 2.0 * U_LO)
    s = np.clip(s, 1e-15, 1 - 1e-15)
    return np.log(s) - np.log1p(-s)


def theta_of_params(p, mu3):
    return theta_of_u(u_of_params(p, mu3))


def active_bounds(u):
    """List of active-bound names for a single u (host side)."""
    u = np.asarray(u)
    out = []
    for i, (lo, hi) in enumerate(BOUND_NAMES):
        if u[i] < ACTIVE_TOL:
            out.append(lo)
        elif u[i] > 1.0 - ACTIVE_TOL:
            out.append(hi)
    return out
