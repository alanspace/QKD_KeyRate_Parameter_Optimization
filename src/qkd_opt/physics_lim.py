"""Key length l and rate R = l/N for decoy-state BB84, Lim et al. 2014 finite-key bounds.

Pure JAX, float64, scalar in (params, L, n_X); use jax.vmap for batches.

params = [mu1, mu2, P_mu1, P_mu2, q_x]; mu3 = Consts.mu3; P_mu3 = 1 - P_mu1 - P_mu2.

Expected-count model (the paper does not print it; this is the standard reading of
"n_X fixed, N <= n_X / R"):
    N         = n_X / (q_x^2 sum_k p_k R_k)
    n_{X,k}   = N q_x^2 p_k R_k,      n_{Z,k} = N (1-q_x)^2 p_k R_k
    m_{X,k}   = N q_x^2 p_k e_k,      m_{Z,k} = N (1-q_x)^2 p_k e_k
    e_obs     = sum_k m_{X,k} / n_X,  lambda_EC = n_X f_EC h(e_obs)

Deliberate, documented departures from a literal transcription (all keep the bound valid):
  * the floor in Eq. (1) is dropped (continuous relaxation for optimisation);
  * s_{X,0}, s_{Z,0} are clamped at >= 0 (true vacuum counts are >= 0);
  * if s_{X,1} <= 0 or s_{Z,1} <= 0 the single-photon term contributes nothing (h(phi)=1);
  * phi_X is capped at 1/2 (h is symmetric; phi >= 1/2 yields no single-photon key).
"""
import functools

import jax
import jax.numpy as jnp
from jax import lax

from .presets import Consts, Flags

jax.config.update("jax_enable_x64", True)

LN2 = jnp.log(2.0)
N_EPS_GRID = 128   # log-grid points used to bracket the greatest eps fixed point


def binary_entropy(x):
    """h(x) for x in [0, 1/2]; exact 0 at x = 0 with finite gradient elsewhere."""
    x = jnp.clip(x, 0.0, 0.5)
    xs = jnp.where(x > 0, x, 0.5)  # safe value for the masked branch
    h = -(xs * jnp.log2(xs) + (1 - xs) * jnp.log2(1 - xs))
    return jnp.where(x > 0, h, 0.0)


def _gamma(a, b, c, d):
    """gamma(a,b,c,d) of Eq. (5). Requires 0 < b < 1, c, d > 0."""
    inner = (c + d) * (1 - b) * b / (c * d * LN2) * jnp.log2((c + d) / (c * d * (1 - b) * b) * 21.0**2 / a**2)
    return jnp.sqrt(jnp.maximum(inner, 0.0))


THETA_BISECT_ITERS = 80


def _dH(y, delta):
    """H(y + delta) - H(y) without cancellation (log1p form), for 0 < y, y + delta < 1."""
    y, delta = jnp.asarray(y), jnp.asarray(delta)
    return (-y * jnp.log1p(delta / y) - delta * jnp.log(y + delta)
            - (1.0 - y) * jnp.log1p(-delta / (1.0 - y)) + delta * jnp.log(1.0 - y - delta)) / jnp.log(2.0)


def _fmc_log2_tail(theta, a, b, c, d):
    """g(theta) = log2 P(theta) - log2 a, with the Fung-Ma-Chau tail bound (arXiv:0910.0312, Eq. 18/B9):
        P < sqrt(N / (c d b (1-b))) * 2^(-N xi(theta)),
        xi(theta) = H(b + theta - q theta) - q H(b) - (1 - q) H(b + theta),  q = c/N,  N = c + d.
    N xi is evaluated as  c * dH(b, d theta/N) - d * dH(b + d theta/N, c theta/N)  (algebraically equal,
    no cancellation of O(1) entropies multiplied by N). g is decreasing in theta on [0, 1-b)."""
    theta, a, b, c, d = (jnp.asarray(x) for x in (theta, a, b, c, d))
    N = c + d
    s = d * theta / N
    Nxi = c * _dH(b, s) - d * _dH(b + s, c * theta / N)
    return 0.5 * jnp.log2(N / (c * d * b * (1.0 - b))) - Nxi - jnp.log2(a)


def _theta_max(b):
    return (1.0 - b) * (1.0 - 1e-12)


def _theta_bisect(a, b, c, d):
    lo, hi = jnp.zeros_like(b), _theta_max(b)

    def body(_, st):
        lo_, hi_ = st
        mid = 0.5 * (lo_ + hi_)
        pos = _fmc_log2_tail(mid, a, b, c, d) > 0      # bound still above a: need larger theta
        return jnp.where(pos, mid, lo_), jnp.where(pos, hi_, mid)

    lo, hi = lax.fori_loop(0, THETA_BISECT_ITERS, body, (lo, hi))
    th = 0.5 * (lo + hi)
    th = jnp.where(_fmc_log2_tail(jnp.zeros_like(b), a, b, c, d) <= 0, 0.0, th)       # bound already <= a
    return jnp.where(_fmc_log2_tail(_theta_max(b), a, b, c, d) > 0, _theta_max(b), th)  # no root: cap


@jax.custom_jvp
def theta_exact(a, b, c, d):
    """Smallest theta with P(theta) <= a (exact Fung-Ma-Chau), by bisection. Differentiable to any order
    through the implicit-function rule d theta = -(dg/dx . dx) / (dg/dtheta) at interior roots."""
    return _theta_bisect(a, b, c, d)


@theta_exact.defjvp
def _theta_exact_jvp(primals, tangents):
    a, b, c, d = primals
    th = theta_exact(a, b, c, d)
    g_th = jax.grad(_fmc_log2_tail, argnums=0)(th, a, b, c, d)
    interior = (th > 0) & (th < _theta_max(b)) & (g_th < 0)
    g_th_safe = jnp.where(interior, g_th, -1.0)
    th_safe = jnp.where(interior, th, 0.5 * _theta_max(b))
    _, g_dot = jax.jvp(lambda a_, b_, c_, d_: _fmc_log2_tail(th_safe, a_, b_, c_, d_), primals, tangents)
    th_max_dot = jax.jvp(_theta_max, (b,), (tangents[1],))[1]
    th_dot = jnp.where(interior, -g_dot / g_th_safe, jnp.where(th > 0, th_max_dot, 0.0))
    return th, th_dot


def _counts(params, L, n_X, c: Consts, f: Flags):
    mu1, mu2, P1, P2, qx = params[0], params[1], params[2], params[3], params[4]
    mu3 = c.mu3
    mu = jnp.stack([mu1, mu2, jnp.asarray(mu3, params.dtype)])
    pk = jnp.stack([P1, P2, 1.0 - P1 - P2])
    eta_ch = 10.0 ** (-c.alpha * L / 10.0)
    eta_sys = c.eta_bob * eta_ch
    D = 1.0 - (1.0 - 2.0 * c.p_dc) * jnp.exp(-eta_sys * mu)
    Rk = D * (1.0 + c.p_ap) if f.afterpulse_in_rate else D
    eta_e = eta_ch if f.emis_eta == "ch" else eta_sys
    ek = c.p_dc + c.e_mis * (1.0 - jnp.exp(-eta_e * mu)) + c.p_ap * D / 2.0
    qz = 1.0 - qx
    N = n_X / (qx**2 * jnp.sum(pk * Rk))
    nXk = N * qx**2 * pk * Rk
    nZk = N * qz**2 * pk * Rk
    mXk = N * qx**2 * pk * ek
    mZk = N * qz**2 * pk * ek
    return dict(mu=mu, pk=pk, N=N, nXk=nXk, nZk=nZk, mXk=mXk, mZk=mZk, eta_ch=eta_ch)


def _length_given_eps(eps, cnt, n_X, c: Consts, f: Flags):
    mu, pk, N = cnt["mu"], cnt["pk"], cnt["N"]
    mu1, mu2, mu3 = mu[0], mu[1], mu[2]
    nXk, nZk, mXk, mZk = cnt["nXk"], cnt["nZk"], cnt["mXk"], cnt["mZk"]
    n_Z = jnp.sum(nZk)
    m_Z = jnp.sum(mZk) if f.mz_mode == "errors" else jnp.sum(nXk)
    ln = jnp.log(21.0 / eps)
    dX = jnp.sqrt(n_X / 2.0 * ln)
    dZ = jnp.sqrt(n_Z / 2.0 * ln)
    dm = jnp.sqrt(m_Z / 2.0 * ln)
    w = jnp.exp(mu) / pk
    lo = (lambda x: jnp.maximum(x, 0.0)) if f.clamp_minus else (lambda x: x)
    nXp, nXm = w * (nXk + dX), w * lo(nXk - dX)
    nZp, nZm = w * (nZk + dZ), w * lo(nZk - dZ)
    mZp, mZm = w * (mZk + dm), w * lo(mZk - dm)

    tau0 = jnp.sum(pk * jnp.exp(-mu))
    tau1 = jnp.sum(pk * jnp.exp(-mu) * mu)
    den1 = mu1 * (mu2 - mu3) - mu2**2 + mu3**2

    def s0(npl, nmi):
        return jnp.maximum(tau0 * (mu2 * nmi[2] - mu3 * npl[1]) / (mu2 - mu3), 0.0)

    def s1(npl, nmi, s_0):
        num = tau1 * mu1 * (nmi[1] - npl[2] - (mu2**2 - mu3**2) / mu1**2 * (npl[0] - s_0 / tau0))
        return num / den1

    sX0 = s0(nXp, nXm)
    sZ0 = s0(nZp, nZm)
    sX1 = s1(nXp, nXm, sX0)
    sZ1 = s1(nZp, nZm, sZ0)
    vZ1 = tau1 * (mZp[1] - mZm[2]) / (mu2 - mu3)

    ok = (sX1 > 0) & (sZ1 > 0)
    sX1s = jnp.where(ok, sX1, 1.0)
    sZ1s = jnp.where(ok, sZ1, 1.0)
    b = vZ1 / sZ1s
    b_ok = ok & (b > 0) & (b < 0.5)
    bs = jnp.where(b_ok, b, 0.25)
    if f.gamma_mode == "printed":
        gam = _gamma(eps, bs, sZ1s, sX1s)
    else:
        th_ex = theta_exact(eps / 21.0, bs, sZ1s, sX1s)
        gam = th_ex if f.gamma_mode == "exact" else jnp.maximum(_gamma(eps, bs, sZ1s, sX1s), th_ex)
    phi = jnp.where(b_ok, jnp.minimum(bs + gam, 0.5), 0.5)
    h_phi = binary_entropy(phi)
    single = jnp.where(ok, sX1s * (1.0 - h_phi), 0.0)

    e_obs = jnp.sum(mXk) / n_X
    lam = n_X * c.f_ec * binary_entropy(jnp.minimum(e_obs, 0.5))
    l = sX0 + single - lam - 6.0 * jnp.log2(21.0 / eps) - jnp.log2(2.0 / c.eps_cor)
    aux = dict(sX0=sX0, sX1=sX1, sZ0=sZ0, sZ1=sZ1, vZ1=vZ1, phi=phi, e_obs=e_obs,
               lambda_ec=lam, n_Z=n_Z, m_Z=m_Z)
    return l, aux


def _solve_eps_value(cnt, n_X, c: Consts, f: Flags):
    """Greatest fixed point of eps = F(eps) := kappa * max(l(eps), 1) (value only).

    F is increasing in eps (l grows with eps) and F(kappa n_X) <= kappa n_X (l <= n_X), so
    iterating from eps0 = kappa n_X decreases monotonically towards the greatest fixed point
    (eps_K >= greatest root). The iteration is slow near the horizon, so it is followed by a
    128-point log-grid scan of g = log F - log eps on [kappa, eps_K] that brackets the largest
    root, and a 60-step bisection inside that bracket.
    Note: eps = kappa (l <= 1) is always a fixed point. When the non-trivial root disappears
    (near the horizon) the greatest fixed point jumps to kappa, so R(p) is DISCONTINUOUS there --
    a property of the eps_sec = kappa * l convention, not a numerical artefact.
    """
    F = lambda e: c.kappa * jnp.maximum(_length_given_eps(e, cnt, n_X, c, f)[0], 1.0)

    def it(e, _):
        return F(e), None

    eps_hi, _ = lax.scan(it, jnp.asarray(c.kappa * n_X, jnp.float64), None, length=f.n_eps_iter)
    grid = jnp.linspace(jnp.log(jnp.asarray(c.kappa, jnp.float64)), jnp.log(eps_hi), N_EPS_GRID)
    g = jax.vmap(lambda le: jnp.log(F(jnp.exp(le))) - le)(grid)
    j = jnp.max(jnp.where(g >= 0, jnp.arange(N_EPS_GRID), 0))
    log_lo = grid[j]
    log_hi = grid[jnp.minimum(j + 1, N_EPS_GRID - 1)]

    def bis(state, _):
        lo, hi = state
        mid = 0.5 * (lo + hi)
        ok = F(jnp.exp(mid)) >= jnp.exp(mid)
        return (jnp.where(ok, mid, lo), jnp.where(ok, hi, mid)), None

    (log_lo, log_hi), _ = lax.scan(bis, (log_lo, log_hi), None, length=60)
    return jnp.exp(log_lo)


def _F_eps(e, params, L, n_X, c, f):
    cnt = _counts(params, L, n_X, c, f)
    return c.kappa * jnp.maximum(_length_given_eps(e, cnt, n_X, c, f)[0], 1.0)


@functools.partial(jax.custom_jvp, nondiff_argnums=(4,))
def _eps_star(params, L, n_X, c, f):
    """eps_sec = greatest fixed point of kappa * max(l(eps), 1), differentiable to any order via the
    implicit-function rule  d eps = (dF/dx . dx) / (1 - dF/deps)  (custom_jvp, so higher derivatives
    are exact, unlike a one-step correction)."""
    return _solve_eps_value(_counts(params, L, n_X, c, f), n_X, c, f)


@_eps_star.defjvp
def _eps_star_jvp(f, primals, tangents):
    params, L, n_X, c = primals
    e = _eps_star(params, L, n_X, c, f)
    dFde = jax.grad(_F_eps, argnums=0)(e, params, L, n_X, c, f)
    _, dF = jax.jvp(lambda p_, L_, n_, c_: _F_eps(e, p_, L_, n_, c_, f), primals, tangents)
    return e, dF / (1.0 - dFde)


def key_length(params, L, n_X, c: Consts, f: Flags):
    """Return dict(l, N, R, eps_sec, eps_residual, ...) for one point."""
    params = jnp.asarray(params, jnp.float64)
    cnt = _counts(params, L, n_X, c, f)
    if f.eps_mode == "fixed":
        eps = jnp.asarray(c.eps_sec, jnp.float64)
        l, aux = _length_given_eps(eps, cnt, n_X, c, f)
        resid = jnp.asarray(0.0)
    else:
        eps = _eps_star(params, L, n_X, c, f)
        l, aux = _length_given_eps(eps, cnt, n_X, c, f)
        resid = jnp.abs(c.kappa * jnp.maximum(l, 1.0) - eps) / eps
    N = cnt["N"]
    return dict(l=l, N=N, R=l / N, eps_sec=eps, eps_residual=resid, **aux)
