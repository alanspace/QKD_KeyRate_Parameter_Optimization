"""JAX-native global + local optimiser for R = l/N over p = [mu1, mu2, P_mu1, P_mu2, q_x].

API
---
optimize(L, n_X, preset, ...)                  one point, cold start (global + local)
optimize_batch(L[P], n_X[P], preset, ...)      many independent points, cold start, vectorised
optimize_sweep(L_grid, nX_list, preset, ...)   continuation along L for several n_X at once, with a
                                               periodic global branch check, backward backfill after
                                               every branch switch, and horizon refinement (l = 0)

Design
------
* Parameters are optimised in an unconstrained space theta (src/qkd_opt/reparam.py), so every
  protocol constraint holds by construction; no penalties.
* Objective modes (per point):
    mode 0: maximise R = l/N   (used where a positive key exists)
    mode 1: maximise l/n_X     (used beyond the horizon). Maximising R there is degenerate:
            R < 0 is pushed to 0^- by q_x -> 0 (N -> infinity). max_p l keeps a meaningful,
            smooth signal and sign(max_p l) = sign(max_p R) because N > 0, so the horizon is
            exactly where max_p l crosses 0.
  A point is solved in mode 1 when the best start has R <= 0; if that solve finds l > 0 it is
  re-solved in mode 0. Neither R nor l is ever clipped.
* The objective is divided by a per-point scale s = |value at the best start| so tolerances are
  relative (R spans 1e-2 .. 1e-9).
* Global stage: scrambled Sobol points in u-space, evaluated with jit+vmap, diverse top-k kept.
* Local stage: damped saddle-free Newton with the exact 5x5 Hessian (jax.hessian), |eigenvalue|
  regularisation, step cap and a vectorised Armijo backtracking line search; vmapped over
  starts and points; float64 throughout.
* Every returned point carries a status string and a certificate (gradient norms, Hessian
  eigenvalues, active bounds, branch id). Non-finite points get NaN params and status
  'nonfinite' -- never bounds.
* Backend-agnostic: no platform is forced. Use JAX_PLATFORMS=cpu on Apple silicon (Metal lacks
  float64); objective.check_x64() raises if float64 is unavailable.
"""
from __future__ import annotations

import functools
import logging
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from scipy.stats import qmc

from . import reparam
from .objective import evaluate
from .presets import Preset

jax.config.update("jax_enable_x64", True)
log = logging.getLogger("qkd_opt")

SOLVER_STATUS = {0: "converged_grad", 1: "converged_ftol", 2: "max_iter", 3: "nonfinite", 4: "no_descent"}
STATUS_CODE = {v: k for k, v in SOLVER_STATUS.items()}
MAX_STEP = 3.0          # max Newton step length in theta (logit) units
N_ALPHA = 24            # line-search step sizes 2^0 .. 2^-23
DEFAULT_GTOL = 1e-9     # on |grad_theta f|_inf, f = -(objective)/s, dimensionless O(1)
DEFAULT_FTOL = 1e-15    # relative stall tolerance (3 consecutive iterations)
SWITCH_RTOL = 1e-9      # global candidate must beat the warm solution by this relative margin
SWITCH_DIST = 1e-3      # ... and differ by this much (inf-norm in u-space) to count as a new branch


# --------------------------------------------------------------------------------------------
# jitted kernels
# --------------------------------------------------------------------------------------------
def _obj_theta(theta, L, n_X, mode, consts, flags):
    out = evaluate(reparam.params_of_theta(theta, consts.mu3), L, n_X, consts, flags)
    return jnp.where(mode > 0, out["l"] / n_X, out["R"])


def _R_l_theta(theta, L, n_X, consts, flags):
    out = evaluate(reparam.params_of_theta(theta, consts.mu3), L, n_X, consts, flags)
    return out["R"], out["l"]


@functools.partial(jax.jit, static_argnames=("flags",))
def eval_shared(theta, L, n_X, consts, flags):
    """(R, l) for a shared set of starts theta [M,5] at P points -> two arrays [P, M]."""
    per_point = lambda l_, n: jax.vmap(lambda t: _R_l_theta(t, l_, n, consts, flags))(theta)
    return jax.vmap(per_point)(L, n_X)


@functools.partial(jax.jit, static_argnames=("flags",))
def eval_starts(theta, L, n_X, consts, flags):
    """(R, l) for per-point starts theta [P,K,5] -> two arrays [P, K]."""
    return jax.vmap(lambda th, l_, n: jax.vmap(lambda t: _R_l_theta(t, l_, n, consts, flags))(th))(theta, L, n_X)


def _newton(theta0, L, n_X, mode, scale, consts, flags, max_iter, gtol, ftol):
    fun = lambda th: -_obj_theta(th, L, n_X, mode, consts, flags) / scale
    vg = jax.value_and_grad(fun)
    hess = jax.hessian(fun)
    alphas = 0.5 ** jnp.arange(N_ALPHA, dtype=jnp.float64)

    f0, g0 = vg(theta0)
    finite0 = jnp.isfinite(f0) & jnp.all(jnp.isfinite(g0))
    st0 = jnp.where(finite0, jnp.where(jnp.max(jnp.abs(g0)) <= gtol, 0, -1), 3)

    def cond(s):
        return (s[4] < 0) & (s[3] < max_iter)

    def body(s):
        th, f, g, it, _st, stall = s
        H = hess(th)
        H = 0.5 * (H + H.T)
        w, V = jnp.linalg.eigh(H)
        wa = jnp.maximum(jnp.abs(w), 1e-10 * jnp.max(jnp.abs(w)) + 1e-14)
        d = -V @ ((V.T @ g) / wa)
        nd = jnp.linalg.norm(d)
        d = d * jnp.minimum(1.0, MAX_STEP / jnp.maximum(nd, 1e-300))
        slope = g @ d
        ft = jax.vmap(lambda a: fun(th + a * d))(alphas)
        ft = jnp.where(jnp.isfinite(ft), ft, jnp.inf)
        armijo = ft <= f + 1e-4 * alphas * slope
        any_ok = jnp.any(armijo)
        i_ok = jnp.argmax(armijo)
        i_best = jnp.argmin(ft)
        a = jnp.where(any_ok, alphas[i_ok], jnp.where(ft[i_best] < f, alphas[i_best], 0.0))
        th_try = th + a * d
        f_try, g_try = vg(th_try)
        moved = a > 0
        th_n = jnp.where(moved, th_try, th)
        f_n = jnp.where(moved, f_try, f)
        g_n = jnp.where(moved, g_try, g)
        stall = jnp.where(f - f_n <= ftol * jnp.maximum(1.0, jnp.abs(f_n)), stall + 1, 0)
        finite = jnp.isfinite(f_n) & jnp.all(jnp.isfinite(g_n))
        st = jnp.where(~finite, 3,
             jnp.where(jnp.max(jnp.abs(g_n)) <= gtol, 0,
             jnp.where(stall >= 3, 1,
             jnp.where(~moved, 4, -1))))
        return th_n, f_n, g_n, it + 1, st, stall

    init = (theta0, f0, g0, jnp.asarray(0), st0, jnp.asarray(0))
    th, f, g, it, st, _ = lax.while_loop(cond, body, init)
    st = jnp.where(st < 0, 2, st)
    return th, f, it, st


@functools.partial(jax.jit, static_argnames=("flags", "max_iter"))
def local_batch(theta0, L, n_X, mode, scale, consts, flags, max_iter=200, gtol=DEFAULT_GTOL, ftol=DEFAULT_FTOL):
    """theta0 [P,K,5]; L, n_X, mode, scale [P]. Returns theta [P,K,5], f, iters, status [P,K]."""
    def per_point(th0, l_, n, m, s):
        return jax.vmap(lambda t: _newton(t, l_, n, m, s, consts, flags, max_iter, gtol, ftol))(th0)
    return jax.vmap(per_point)(theta0, L, n_X, mode, scale)


@functools.partial(jax.jit, static_argnames=("flags",))
def certify_batch(theta, L, n_X, mode, scale, consts, flags):
    """Certificate at theta [P,5] for the objective actually optimised (mode)."""
    def one(th, l_, n, m, s):
        fun = lambda t: -_obj_theta(t, l_, n, m, consts, flags) / s
        g = jax.grad(fun)(th)
        H = jax.hessian(fun)(th)
        eig = jnp.linalg.eigvalsh(0.5 * (H + H.T))
        p = reparam.params_of_theta(th, consts.mu3)
        out = evaluate(p, l_, n, consts, flags)
        gp = jax.grad(lambda pp: evaluate(pp, l_, n, consts, flags)["R"])(p)
        return dict(params=p, u=reparam.u_of_theta(th), R=out["R"], l=out["l"], N=out["N"],
                    eps_sec=out["eps_sec"], eps_residual=out["eps_residual"],
                    grad_theta=g, hess_eig=eig, grad_p_R=gp)
    return jax.vmap(one)(theta, L, n_X, mode, scale)


# --------------------------------------------------------------------------------------------
# host-side helpers
# --------------------------------------------------------------------------------------------
def sobol_theta(n: int, seed: int) -> np.ndarray:
    m = int(np.ceil(np.log2(max(n, 2))))
    u = qmc.Sobol(d=5, scramble=True, seed=seed).random_base2(m)[:n]
    return reparam.theta_of_u(u)


def _diverse_top_k(score, u_set, k, min_sep=0.05):
    score = np.where(np.isfinite(score), score, -np.inf)
    order = np.argsort(-score, kind="stable")
    chosen = []
    for idx in order[: max(50 * k, k)]:
        if not np.isfinite(score[idx]):
            break
        if all(np.max(np.abs(u_set[idx] - u_set[j])) > min_sep for j in chosen):
            chosen.append(int(idx))
        if len(chosen) == k:
            break
    for idx in order:  # fill up if the diversity filter left gaps
        if len(chosen) == k:
            break
        if int(idx) not in chosen:
            chosen.append(int(idx))
    return np.array(chosen[:k])


def _modes_and_scores(R, l, n_X):
    """mode 1 (maximise l/n_X) where no start has R > 0."""
    Rm = np.where(np.isfinite(R), R, -np.inf)
    mode = (np.max(Rm, axis=-1) <= 0).astype(np.int64)
    score = np.where(mode[:, None] > 0, l / n_X[:, None], R)
    return mode, np.where(np.isfinite(score), score, -np.inf)


def _scale_from(v):
    v = np.asarray(v, np.float64)
    return np.maximum(np.abs(np.where(np.isfinite(v), v, 0.0)), 1e-300)


def global_stage(L, n_X, preset: Preset, n_sobol=4096, top_k=8, seed=0, chunk=8192):
    """Candidate thetas [P,K,5] plus R, l of every candidate [P,K]."""
    L = np.atleast_1d(np.asarray(L, np.float64))
    n_X = np.atleast_1d(np.asarray(n_X, np.float64))
    th_all = sobol_theta(n_sobol, seed)
    u_all = np.asarray(reparam.u_of_theta(jnp.asarray(th_all)))
    chunk = min(chunk, len(th_all))
    R_all = np.empty((len(L), len(th_all)))
    l_all = np.empty_like(R_all)
    for s in range(0, len(th_all), chunk):
        blk = th_all[s:s + chunk]
        n_real = len(blk)
        if n_real < chunk:
            blk = np.concatenate([blk, np.repeat(blk[-1:], chunk - n_real, 0)])
        r, l_ = eval_shared(jnp.asarray(blk), jnp.asarray(L), jnp.asarray(n_X), preset.consts, preset.flags)
        R_all[:, s:s + n_real] = np.asarray(r)[:, :n_real]
        l_all[:, s:s + n_real] = np.asarray(l_)[:, :n_real]
    _, score = _modes_and_scores(R_all, l_all, n_X)
    idx = np.stack([_diverse_top_k(score[i], u_all, top_k) for i in range(len(L))])
    return th_all[idx], np.take_along_axis(R_all, idx, 1), np.take_along_axis(l_all, idx, 1)


def _solve(theta0, L, n_X, preset, max_iter, gtol, ftol, mode=None):
    """Local solve from starts theta0 [P,K,5], choosing the objective mode automatically and
    switching mode 1 -> 0 if a positive key is found. Returns best per point."""
    theta0 = np.asarray(theta0, np.float64)
    P = theta0.shape[0]
    R0, l0 = map(np.asarray, eval_starts(jnp.asarray(theta0), jnp.asarray(L), jnp.asarray(n_X),
                                         preset.consts, preset.flags))
    m_auto, score0 = _modes_and_scores(R0, l0, n_X)
    mode = m_auto if mode is None else np.asarray(mode)
    scale = _scale_from(np.max(score0, axis=1))
    th, f, it, st = map(np.asarray, local_batch(jnp.asarray(theta0), jnp.asarray(L), jnp.asarray(n_X),
                                                jnp.asarray(mode), jnp.asarray(scale), preset.consts,
                                                preset.flags, max_iter=max_iter, gtol=gtol, ftol=ftol))
    val = np.where(np.isfinite(f), -f * scale[:, None], -np.inf)
    best = np.argmax(val, axis=1)
    rows = np.arange(P)
    vbest = val[rows, best]
    hits = np.sum(np.abs(val - vbest[:, None]) <= 1e-9 * np.abs(vbest[:, None]), axis=1)
    out = dict(theta=th[rows, best], iters=it[rows, best], status=st[rows, best], mode=mode.copy(),
               scale=scale, hits=hits, best_start=best)
    # mode 1 that reached l > 0: re-solve in mode 0 from there
    flip = (mode > 0) & (vbest > 0)
    if flip.any():
        idx = np.where(flip)[0]
        sub = _solve(out["theta"][idx][:, None, :], L[idx], n_X[idx], preset, max_iter, gtol, ftol,
                     mode=np.zeros(len(idx), np.int64))
        for k_, i in enumerate(idx):
            for key in ("theta", "iters", "status", "mode", "scale"):
                out[key][i] = sub[key][k_]
    return out


def _finalise(sol, L, n_X, preset):
    cert = certify_batch(jnp.asarray(sol["theta"]), jnp.asarray(L), jnp.asarray(n_X),
                         jnp.asarray(sol["mode"]), jnp.asarray(sol["scale"]), preset.consts, preset.flags)
    cert = {k: np.asarray(v) for k, v in cert.items()}
    P = len(L)
    status, active = [], []
    params = cert["params"].copy()
    for i in range(P):
        fin = np.isfinite(cert["R"][i]) and np.isfinite(cert["l"][i]) and np.all(np.isfinite(cert["grad_theta"][i]))
        s = SOLVER_STATUS[int(sol["status"][i])] if fin else "nonfinite"
        if fin and cert["l"][i] <= 0:
            s += "|beyond_horizon"
        if not fin:
            params[i] = np.nan
        status.append(s)
        active.append(reparam.active_bounds(cert["u"][i]) if fin else [])
    out = dict(
        preset=preset.name, L=np.asarray(L, np.float64), n_X=np.asarray(n_X, np.float64),
        theta=np.asarray(sol["theta"]), params=params, R=cert["R"], l=cert["l"], N=cert["N"],
        eps_sec=cert["eps_sec"], eps_residual=cert["eps_residual"],
        status=np.array(status, dtype=object), solver_iters=np.asarray(sol["iters"]),
        objective_mode=np.where(np.asarray(sol["mode"]) > 0, "l/n_X", "R").astype(object),
        grad_norm_theta=np.max(np.abs(cert["grad_theta"]), axis=1),
        grad_p_R=cert["grad_p_R"], hess_eig=cert["hess_eig"],
        active_bounds=np.array(active + [None], dtype=object)[:-1], scale=np.asarray(sol["scale"]),
    )
    if "hits" in sol:
        out["n_basin_hits"] = np.asarray(sol["hits"])
    return out


# --------------------------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------------------------
def optimize_batch(L, n_X, preset: Preset, n_sobol=4096, top_k=8, seed=0, extra_starts=None,
                   max_iter=200, gtol=DEFAULT_GTOL, ftol=DEFAULT_FTOL):
    """Cold-start optimisation of P independent points. extra_starts: params [P,E,5] (optional)."""
    L = np.atleast_1d(np.asarray(L, np.float64))
    n_X = np.atleast_1d(np.asarray(n_X, np.float64))
    if n_X.size == 1 and L.size > 1:
        n_X = np.full_like(L, n_X[0])
    if L.size == 1 and n_X.size > 1:
        L = np.full_like(n_X, L[0])
    th_c, _, _ = global_stage(L, n_X, preset, n_sobol, top_k, seed)
    if extra_starts is not None:
        th_e = reparam.theta_of_params(np.asarray(extra_starts), preset.consts.mu3)
        th_c = np.concatenate([th_c, th_e], axis=1)
    sol = _solve(th_c, L, n_X, preset, max_iter, gtol, ftol)
    out = _finalise(sol, L, n_X, preset)
    out["best_start"] = sol["best_start"]
    out["n_starts"] = th_c.shape[1]
    return out


def optimize(L, n_X, preset: Preset, **kw):
    """Single point, cold start. Returns a dict of per-point values."""
    out = optimize_batch([L], [n_X], preset, **kw)
    return {k: (v[0] if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) == 1 else v) for k, v in out.items()}


def warm_solve(theta, L, n_X, preset, max_iter=200, gtol=DEFAULT_GTOL, ftol=DEFAULT_FTOL):
    """Local solve from one start per point, theta [P,5]."""
    return _solve(np.asarray(theta)[:, None, :], np.asarray(L, np.float64), np.asarray(n_X, np.float64),
                  preset, max_iter, gtol, ftol)


def _value(R, l, n_X, mode):
    return l / n_X if mode > 0 else R


def optimize_sweep(L_grid, nX_list, preset: Preset, global_every=10, n_sobol_init=16384, top_k_init=16,
                   n_sobol_check=2048, top_k_check=4, seed=0, max_iter=200, gtol=DEFAULT_GTOL,
                   ftol=DEFAULT_FTOL, refine_horizon=True, horizon_iters=30, branch_check=True):
    """Continuation along L for every n_X in nX_list (chains advance together, vectorised).

    Returns dict with per-point arrays shaped [n_nX, n_L], 'events' (branch switches, global
    refinements, backfills, fallbacks) and 'horizon' (per n_X, km, sign change of max_p l).
    """
    t0 = time.perf_counter()
    L_grid = np.asarray(L_grid, np.float64)
    nX = np.asarray(nX_list, np.float64)
    P, nL = len(nX), len(L_grid)
    events = []
    theta = np.zeros((P, nL, 5))
    st_arr = np.zeros((P, nL), int)
    it_arr = np.zeros((P, nL), int)
    source = np.empty((P, nL), dtype=object)
    branch_id = np.zeros((P, nL), int)
    branch = np.zeros(P, int)

    c0 = optimize_batch(np.full(P, L_grid[0]), nX, preset, n_sobol=n_sobol_init, top_k=top_k_init,
                        seed=seed, max_iter=max_iter, gtol=gtol, ftol=ftol)
    theta[:, 0] = c0["theta"]
    st_arr[:, 0] = [STATUS_CODE.get(s.split("|")[0], 3) for s in c0["status"]]
    it_arr[:, 0] = c0["solver_iters"]
    source[:, 0] = "cold"
    last_check = 0

    def RL(th, L):
        R, l = eval_starts(jnp.asarray(th)[:, None, :], jnp.asarray(L), jnp.asarray(nX), preset.consts, preset.flags)
        return np.asarray(R)[:, 0], np.asarray(l)[:, 0]

    for j in range(1, nL):
        L = np.full(P, L_grid[j])
        w = warm_solve(theta[:, j - 1], L, nX, preset, max_iter, gtol, ftol)
        th, st, it = w["theta"].copy(), w["status"].copy(), w["iters"].copy()
        R, l = RL(th, L)
        src = np.array(["warm"] * P, dtype=object)
        bad = (st == 3) | (st == 2) | ~np.isfinite(R)
        do_check = branch_check and (j % global_every == 0 or j == nL - 1)
        idx = np.arange(P) if do_check else np.where(bad)[0]
        if len(idx):
            g = optimize_batch(L[idx], nX[idx], preset,
                               n_sobol=n_sobol_check if do_check else n_sobol_init,
                               top_k=top_k_check if do_check else top_k_init,
                               seed=seed + j, max_iter=max_iter, gtol=gtol, ftol=ftol)
            for a, i in enumerate(idx):
                mode = 1 if (g["R"][a] <= 0 and R[i] <= 0) else 0
                vg_ = _value(g["R"][a], g["l"][a], nX[i], mode)
                vw_ = _value(R[i], l[i], nX[i], mode) if np.isfinite(R[i]) else -np.inf
                if bad[i]:
                    events.append(dict(type="fallback", n_X=float(nX[i]), L=float(L_grid[j]),
                                       warm_status=SOLVER_STATUS[int(st[i])], warm_value=float(vw_),
                                       cold_value=float(vg_)))
                    log.warning("fallback n_X=%.0e L=%.3f warm status=%s", nX[i], L_grid[j], SOLVER_STATUS[int(st[i])])
                if np.isfinite(vg_) and (vg_ > vw_ + SWITCH_RTOL * abs(vw_) or bad[i]):
                    u_w = np.asarray(reparam.u_of_theta(jnp.asarray(th[i])))
                    u_g = np.asarray(reparam.u_of_theta(jnp.asarray(g["theta"][a])))
                    dist = float(np.max(np.abs(u_w - u_g)))
                    kind = "fallback_cold" if bad[i] else ("branch_switch" if dist > SWITCH_DIST else "global_refine")
                    if kind == "branch_switch":
                        branch[i] += 1
                    if kind != "fallback_cold":
                        events.append(dict(
                            type=kind, n_X=float(nX[i]), L=float(L_grid[j]), objective="l/n_X" if mode else "R",
                            warm_value=float(vw_), global_value=float(vg_),
                            rel_gain=float((vg_ - vw_) / abs(vw_)) if vw_ not in (0, -np.inf) else float("inf"),
                            u_dist=dist, new_branch=int(branch[i]),
                            params_warm=np.asarray(reparam.params_of_theta(jnp.asarray(th[i]), preset.consts.mu3)).tolist(),
                            params_global=np.asarray(g["params"][a]).tolist()))
                        log.info("%s n_X=%.0e L=%.3f warm=%.6e global=%.6e dist=%.3g",
                                 kind, nX[i], L_grid[j], vw_, vg_, dist)
                    th[i] = g["theta"][a]
                    st[i] = STATUS_CODE.get(g["status"][a].split("|")[0], 3)
                    it[i] = g["solver_iters"][a]
                    src[i] = kind
                    if kind == "branch_switch":
                        _backfill(i, j, last_check, theta, st_arr, it_arr, source, branch_id, branch,
                                  th[i], L_grid, nX, preset, events, max_iter, gtol, ftol)
            if do_check:
                last_check = j
        theta[:, j], st_arr[:, j], it_arr[:, j], source[:, j] = th, st, it, src
        branch_id[:, j] = branch

    flat_L = np.tile(L_grid, P)
    flat_nX = np.repeat(nX, nL)
    R_flat, l_flat = map(np.asarray, eval_starts(jnp.asarray(theta.reshape(-1, 1, 5)), jnp.asarray(flat_L),
                                                 jnp.asarray(flat_nX), preset.consts, preset.flags))
    mode = (R_flat[:, 0] <= 0).astype(np.int64)
    val = np.where(mode > 0, l_flat[:, 0] / flat_nX, R_flat[:, 0])
    sol = dict(theta=theta.reshape(-1, 5), status=st_arr.ravel(), iters=it_arr.ravel(), mode=mode,
               scale=_scale_from(val))
    fin = _finalise(sol, flat_L, flat_nX, preset)
    out = {k: (v.reshape(P, nL, *v.shape[1:]) if isinstance(v, np.ndarray) and v.shape[:1] == (P * nL,) else v)
           for k, v in fin.items()}
    out["source"] = source
    out["branch_id"] = branch_id
    out["events"] = events
    out["L_grid"], out["nX_list"] = L_grid, nX
    if refine_horizon:
        out["horizon"] = [_horizon(out, i, preset, horizon_iters, max_iter, gtol, ftol) for i in range(P)]
    out["wall_time_s"] = time.perf_counter() - t0
    return out


def _backfill(i, j, last_check, theta, st_arr, it_arr, source, branch_id, branch, th_new, L_grid, nX,
              preset, events, max_iter, gtol, ftol):
    """After a branch switch detected at grid index j, walk back towards the previous check and
    replace stored points while the new branch is better; this locates the crossover."""
    th = th_new.copy()
    n_rep = 0
    for k in range(j - 1, last_check, -1):
        Lk = np.array([L_grid[k]])
        w = warm_solve(th[None], Lk, nX[i:i + 1], preset, max_iter, gtol, ftol)
        Rn, ln = map(np.asarray, eval_starts(jnp.asarray(w["theta"])[:, None, :], jnp.asarray(Lk),
                                             jnp.asarray(nX[i:i + 1]), preset.consts, preset.flags))
        Ro, lo = map(np.asarray, eval_starts(jnp.asarray(theta[i, k])[None, None, :], jnp.asarray(Lk),
                                             jnp.asarray(nX[i:i + 1]), preset.consts, preset.flags))
        mode = 1 if (Rn[0, 0] <= 0 and Ro[0, 0] <= 0) else 0
        vn = _value(Rn[0, 0], ln[0, 0], nX[i], mode)
        vo = _value(Ro[0, 0], lo[0, 0], nX[i], mode)
        if np.isfinite(vn) and vn > vo + SWITCH_RTOL * abs(vo):
            theta[i, k] = w["theta"][0]
            st_arr[i, k], it_arr[i, k] = w["status"][0], w["iters"][0]
            source[i, k], branch_id[i, k] = "backfill", branch[i]
            th = w["theta"][0]
            n_rep += 1
        else:
            break
    events.append(dict(type="backfill", n_X=float(nX[i]), L_switch=float(L_grid[j]), n_replaced=n_rep,
                       crossover_between_km=[float(L_grid[j - n_rep - 1]), float(L_grid[j - n_rep])]))


def _horizon(out, i, preset, iters, max_iter, gtol, ftol):
    """Largest L with max_p l > 0: bisection between the last grid point with l > 0 and the next,
    warm-started from the optimum at the lower bracket end."""
    L_grid, nX = out["L_grid"], out["nX_list"][i]
    l = out["l"][i]
    pos = np.where(l > 0)[0]
    if len(pos) == 0:
        return dict(n_X=float(nX), horizon_km=None, note="l <= 0 on the whole grid")
    k = pos[-1]
    if k == len(L_grid) - 1:
        return dict(n_X=float(nX), horizon_km=None, note=f"l > 0 up to the grid end {L_grid[-1]} km")
    lo, hi = float(L_grid[k]), float(L_grid[k + 1])
    th = out["theta"][i, k]
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        w = warm_solve(th[None], np.array([mid]), np.array([nX]), preset, max_iter, gtol, ftol)
        R, _l = eval_starts(jnp.asarray(w["theta"])[:, None, :], jnp.asarray([mid]), jnp.asarray([nX]),
                            preset.consts, preset.flags)
        if float(np.asarray(_l)[0, 0]) > 0:
            lo, th = mid, w["theta"][0]
        else:
            hi = mid
    return dict(n_X=float(nX), horizon_km=lo, bracket_km=[lo, hi],
                note="bisection on sign of optimised l, warm-started; non-monotone l(L) between grid points would be missed")
