"""Single entry point for the physics: evaluate(params, L, n_X, consts, flags) -> dict(l, N, R, ...).

* flags.model == "lim":        src/qkd_opt/physics_lim.py
* flags.model == "legacy_src": src/qkd/model.py called UNMODIFIED, cross-check only (the physics in src/qkd is
  frozen by project rule); N is reconstructed with the same formula src/qkd uses internally.

No clipping of R or l is performed here: beyond the horizon l < 0 and R < 0 are returned as-is,
so optimizers keep a gradient signal.
"""
import jax
import jax.numpy as jnp

from .presets import Consts, Flags
from . import physics_lim

jax.config.update("jax_enable_x64", True)


def _legacy(params, L, n_X, c: Consts):
    from src.qkd.model import calculate_key_rates_and_metrics  # imported lazily: heavy module

    out = calculate_key_rates_and_metrics(
        params, L, n_X, c.alpha, c.eta_bob, c.p_dc, c.eps_sec, c.eps_cor, c.f_ec, c.e_mis, c.p_ap, 1)
    R, l = out[0], out[-1]
    mu1, mu2, P1, P2, qx = params[0], params[1], params[2], params[3], params[4]
    mu = jnp.stack([mu1, mu2, jnp.asarray(c.mu3, params.dtype)])
    pk = jnp.stack([P1, P2, 1.0 - P1 - P2])
    eta_sys = c.eta_bob * 10.0 ** (-c.alpha * L / 10.0)
    D = 1.0 - (1.0 - 2.0 * c.p_dc) * jnp.exp(-eta_sys * mu)
    N = n_X / (qx**2 * jnp.sum(pk * D))       # src/qkd/physics.py:calculate_N with n_X_total == n_X
    return dict(l=l, N=N, R=R, eps_sec=jnp.asarray(c.eps_sec), eps_residual=jnp.asarray(0.0))


def evaluate(params, L, n_X, consts: Consts, flags: Flags):
    params = jnp.asarray(params, jnp.float64)
    L = jnp.asarray(L, jnp.float64)
    n_X = jnp.asarray(n_X, jnp.float64)
    if flags.model == "lim":
        return physics_lim.key_length(params, L, n_X, consts, flags)
    if flags.model == "legacy_src":
        if isinstance(consts.mu3, (int, float)) and consts.mu3 != 2e-4:
            raise ValueError("legacy src/qkd hard-codes mu3 = 2e-4")
        return _legacy(params, L, n_X, consts)
    raise ValueError(f"unknown model {flags.model!r}")


def key_rate(params, L, n_X, consts: Consts, flags: Flags):
    return evaluate(params, L, n_X, consts, flags)["R"]


def check_x64():
    """Raise if float64 is not actually in use (e.g. Apple Metal backend)."""
    x = jnp.ones(()) * 1.0
    if x.dtype != jnp.float64:
        raise RuntimeError("float64 is required; jax_enable_x64 is off or the backend lacks float64")
    backend = jax.default_backend()
    if backend.lower() == "metal":
        raise RuntimeError("Apple Metal backend has no float64; run with JAX_PLATFORMS=cpu")
    return backend
