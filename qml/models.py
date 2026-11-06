"""Surrogate models (.venv-qml only; no physics import). PREREGISTRATION.md section 5.

All trainable models map x' = pi (x - 1/2), x in {u_L, v}, to z in [-1, 1]^5. The fixed affine map z -> u
(section 6) is applied outside the models, identically for all.

  PQC-full     data re-uploading, per layer and qubit: RX(a1 u_L') RY(a2 v') (a trainable, init 1),
               Rot(phi, theta, omega), then a CZ ring; outputs <Z_i>, i = 0..4. 5 n L parameters.
  PQC-product  the same without the CZ ring.
  NN           2 -> h -> h -> 5 MLP, tanh hidden and output. h^2 + 9h + 5 parameters.
  RFF ridge    [1, cos(w.x'), sin(w.x')] with w = sum_{l,q} c_lq (a_lq1, a_lq2), c iid uniform {-1, 0, 1},
               sampled from a trained PQC-full's input scalings; closed-form ridge to z; z clipped to [-1, 1].
  lookup-2D    per TRAIN row linear in u_L over the row's training nodes, then linear in v between the two
               neighbouring rows; constant outside a row's u_L range. Interpolates u directly.
  lookup-nX    linear in L over the training nodes of that n_X (TRAIN n_X only).
"""
from __future__ import annotations

import math
import time

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

PQC_GRID = [(5, 3), (6, 3), (8, 3), (5, 6), (6, 6), (8, 6)]      # (n_qubits, layers), ordered by 5 n L = 75..240
NN_WIDTHS = [5, 6, 7, 8, 10, 12]                                  # matched in order, h^2 + 9h + 5
N_OUT = 5


def xprime(uL, v):
    return jnp.pi * (jnp.stack([jnp.asarray(uL), jnp.asarray(v)], -1) - 0.5)


# ---------------------------------------------------------------- PQC
def make_pqc(n_qubits: int, layers: int, entangle: bool):
    import pennylane as qml
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="jax", diff_method="backprop")
    def circuit(w, x):
        for l_ in range(layers):
            for q in range(n_qubits):
                qml.RX(w["a"][l_, q, 0] * x[0], wires=q)
                qml.RY(w["a"][l_, q, 1] * x[1], wires=q)
                qml.Rot(w["r"][l_, q, 0], w["r"][l_, q, 1], w["r"][l_, q, 2], wires=q)
            if entangle and n_qubits > 1:
                for q in range(n_qubits):
                    qml.CZ(wires=[q, (q + 1) % n_qubits])
        return [qml.expval(qml.PauliZ(i)) for i in range(N_OUT)]

    def init(seed):
        k = jax.random.PRNGKey(seed)
        return dict(a=jnp.ones((layers, n_qubits, 2)),
                    r=jax.random.uniform(k, (layers, n_qubits, 3), minval=-math.pi, maxval=math.pi))

    def apply(w, X):
        return jax.vmap(lambda x: jnp.stack(circuit(w, x)))(X)

    return init, apply, 5 * n_qubits * layers


# ---------------------------------------------------------------- NN
def make_nn(h: int):
    def init(seed):
        ks = jax.random.split(jax.random.PRNGKey(seed), 3)
        g = lambda k, a, b: jax.random.normal(k, (a, b)) * jnp.sqrt(1.0 / a)
        return dict(W1=g(ks[0], 2, h), b1=jnp.zeros(h), W2=g(ks[1], h, h), b2=jnp.zeros(h),
                    W3=g(ks[2], h, N_OUT), b3=jnp.zeros(N_OUT))

    def apply(w, X):
        a = jnp.tanh(X @ w["W1"] + w["b1"])
        a = jnp.tanh(a @ w["W2"] + w["b2"])
        return jnp.tanh(a @ w["W3"] + w["b3"])

    return init, apply, h * h + 9 * h + 5


def n_params(w):
    return int(sum(np.size(x) for x in jax.tree_util.tree_leaves(w)))


# ---------------------------------------------------------------- training (shared loop)
def cosine_lr(step, total, lr0=1e-2, lr1=1e-4):
    return lr1 + 0.5 * (lr0 - lr1) * (1 + jnp.cos(jnp.pi * step / total))


def make_trainer(apply, total_steps):
    def loss(w, X, Z):
        return jnp.mean((apply(w, X) - Z) ** 2)

    @jax.jit
    def step(w, m, v, t, X, Z):
        l_, g = jax.value_and_grad(loss)(w, X, Z)
        b1, b2, eps = 0.9, 0.999, 1e-8
        lr = cosine_lr(t, total_steps)
        m = jax.tree_util.tree_map(lambda m_, g_: b1 * m_ + (1 - b1) * g_, m, g)
        v = jax.tree_util.tree_map(lambda v_, g_: b2 * v_ + (1 - b2) * g_ * g_, v, g)
        mh = jax.tree_util.tree_map(lambda m_: m_ / (1 - b1 ** (t + 1)), m)
        vh = jax.tree_util.tree_map(lambda v_: v_ / (1 - b2 ** (t + 1)), v)
        w = jax.tree_util.tree_map(lambda w_, a_, b_: w_ - lr * a_ / (jnp.sqrt(b_) + eps), w, mh, vh)
        return w, m, v, l_

    def zeros(w):
        return jax.tree_util.tree_map(jnp.zeros_like, w)

    return step, zeros, jax.jit(loss)


# ---------------------------------------------------------------- RFF ridge on the PQC spectrum
def sample_frequencies(a, D2, seed):
    """a: trained PQC input scalings [layers, n, 2]; returns D2 frequencies [D2, 2]."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a).reshape(-1, 2)
    c = rng.integers(-1, 2, size=(D2, a.shape[0]))
    return c @ a


def rff_features(X, W):
    P = np.asarray(X) @ np.asarray(W).T
    return np.concatenate([np.ones((len(P), 1)), np.cos(P), np.sin(P)], axis=1)


def fit_ridge(F, Z, lam):
    A = F.T @ F + lam * np.eye(F.shape[1])
    return np.linalg.solve(A, F.T @ Z)


def ridge_D2_for(pqc_params: int):
    """5 (D + 1) ~ pqc_params with D = 2 D2 features (+1 bias)."""
    return max(1, round((pqc_params / 5 - 1) / 2))


# ---------------------------------------------------------------- lookup tables (interpolate u directly)
def lookup_2d(train_v, train_uL, train_u, query_v, query_uL):
    rows = np.unique(train_v)
    tabs = []
    for r in rows:
        m = train_v == r
        o = np.argsort(train_uL[m])
        tabs.append((train_uL[m][o], train_u[m][o]))

    def row_interp(i, x):
        xs, us = tabs[i]
        return np.stack([np.interp(x, xs, us[:, c]) for c in range(us.shape[1])], -1)

    out = np.empty((len(query_v), train_u.shape[1]))
    j = np.clip(np.searchsorted(rows, query_v, side="right") - 1, 0, len(rows) - 2)
    for jj in np.unique(j):
        m = j == jj
        w = np.clip((query_v[m] - rows[jj]) / (rows[jj + 1] - rows[jj]), 0.0, 1.0)[:, None]
        out[m] = (1 - w) * row_interp(jj, query_uL[m]) + w * row_interp(jj + 1, query_uL[m])
    return out, sum(t[0].size for t in tabs)


def lookup_nx(train_nx, train_L, train_u, query_nx, query_L):
    out = np.full((len(query_nx), train_u.shape[1]), np.nan)
    size = 0
    for nx in np.unique(train_nx):
        m = train_nx == nx
        o = np.argsort(train_L[m])
        xs, us = train_L[m][o], train_u[m][o]
        size += xs.size
        q = query_nx == nx
        if q.any():
            out[q] = np.stack([np.interp(query_L[q], xs, us[:, c]) for c in range(us.shape[1])], -1)
    return out, size


def timed(fn, *a, reps=3):
    """(result, first-call seconds, median steady seconds)."""
    t = time.perf_counter(); r = jax.block_until_ready(fn(*a)); first = time.perf_counter() - t
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); r = jax.block_until_ready(fn(*a)); ts.append(time.perf_counter() - t)
    return r, first, float(np.median(ts))
