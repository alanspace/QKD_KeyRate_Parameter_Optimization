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
    phi = jnp.where(b_ok, jnp.minimum(bs + _gamma(eps, bs, sZ1s, sX1s), 0.5), 0.5)
    h_phi = binary_entropy(phi)
    single = jnp.where(ok, sX1s * (1.0 - h_phi), 0.0)

    e_obs = jnp.sum(mXk) / n_X
    lam = n_X * c.f_ec * binary_entropy(jnp.minimum(e_obs, 0.5))
    l = sX0 + single - lam - 6.0 * jnp.log2(21.0 / eps) - jnp.log2(2.0 / c.eps_cor)
    aux = dict(sX0=sX0, sX1=sX1, sZ0=sZ0, sZ1=sZ1, vZ1=vZ1, phi=phi, e_obs=e_obs,
               lambda_ec=lam, n_Z=n_Z, m_Z=m_Z)
    return l, aux


def _solve_eps(cnt, n_X, c: Consts, f: Flags):
    """Greatest fixed point of eps = F(eps) := kappa * max(l(eps), 1), differentiable in params.

    F is increasing in eps (l grows with eps) and F(kappa n_X) <= kappa n_X (l <= n_X), so
    iterating from eps0 = kappa n_X decreases monotonically towards the greatest fixed point
    (eps_K >= greatest root). The iteration is slow near the horizon, so it is followed by a
    128-point log-grid scan of g = log F - log eps on [kappa, eps_K] that brackets the largest
    root, and a 60-step bisection inside that bracket.
    Note: eps = kappa (l <= 1) is always a fixed point. When the non-trivial root disappears
    (near the horizon) the greatest fixed point jumps to kappa, so R(p) is DISCONTINUOUS there --
    a property of the eps_sec = kappa * l convention, not a numerical artefact. Both stages run on stop_gradient'ed counts; one implicit-
    function correction step then restores the exact first derivative
        d eps / dp = kappa dl/dp / (1 - kappa dl/deps)
    while leaving the value unchanged at the fixed point.
    """
    cnt_sg = jax.tree_util.tree_map(lax.stop_gradient, cnt)
    F = lambda e, cn: c.kappa * jnp.maximum(_length_given_eps(e, cn, n_X, c, f)[0], 1.0)

    def it(e, _):
        return F(e, cnt_sg), None

    eps_hi, _ = lax.scan(it, jnp.asarray(c.kappa * n_X, jnp.float64), None, length=f.n_eps_iter)
    # g(eps) = log F(eps) - log eps. eps = kappa is always a (trivial) root with g >= 0, so a
    # plain bisection on [kappa, eps_hi] can land on it. Scan a log grid first and bracket the
    # LARGEST grid point with g >= 0; the greatest root lies in (grid[j], grid[j+1]].
    grid = jnp.linspace(jnp.log(jnp.asarray(c.kappa, jnp.float64)), jnp.log(eps_hi), N_EPS_GRID)
    g = jax.vmap(lambda le: jnp.log(F(jnp.exp(le), cnt_sg)) - le)(grid)
    j = jnp.max(jnp.where(g >= 0, jnp.arange(N_EPS_GRID), 0))
    log_lo = grid[j]
    log_hi = grid[jnp.minimum(j + 1, N_EPS_GRID - 1)]

    def bis(state, _):
        lo, hi = state
        mid = 0.5 * (lo + hi)
        ok = F(jnp.exp(mid), cnt_sg) >= jnp.exp(mid)
        return (jnp.where(ok, mid, lo), jnp.where(ok, hi, mid)), None

    (log_lo, log_hi), _ = lax.scan(bis, (log_lo, log_hi), None, length=60)
    e_star = lax.stop_gradient(jnp.exp(log_lo))
    # implicit-function correction (value unchanged at the fixed point, exact first derivative)
    dF = jax.grad(lambda e: F(e, cnt))(e_star)
    return e_star + (F(e_star, cnt) - e_star) / (1.0 - lax.stop_gradient(dF))


def key_length(params, L, n_X, c: Consts, f: Flags):
    """Return dict(l, N, R, eps_sec, eps_residual, ...) for one point."""
    params = jnp.asarray(params, jnp.float64)
    cnt = _counts(params, L, n_X, c, f)
    if f.eps_mode == "fixed":
        eps = jnp.asarray(c.eps_sec, jnp.float64)
        l, aux = _length_given_eps(eps, cnt, n_X, c, f)
        resid = jnp.asarray(0.0)
    else:
        eps = _solve_eps(cnt, n_X, c, f)
        l, aux = _length_given_eps(eps, cnt, n_X, c, f)
        resid = jnp.abs(c.kappa * jnp.maximum(l, 1.0) - eps) / eps
    N = cnt["N"]
    return dict(l=l, N=N, R=l / N, eps_sec=eps, eps_residual=resid, **aux)
