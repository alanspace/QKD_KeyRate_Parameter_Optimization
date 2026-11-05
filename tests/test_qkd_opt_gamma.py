"""gamma_mode tests: exact Fung-Ma-Chau theta vs scipy brentq, higher-order gradients, printed unchanged."""
import itertools
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.test_util import check_grads
from scipy.optimize import brentq

from src.qkd_opt import LEGACY, LIM, LIM_LITERAL, LIM_TIGHT, evaluate, with_flags
from src.qkd_opt.physics_lim import theta_exact

jax.config.update("jax_enable_x64", True)


def _dH_np(y, delta):
    return (-y * np.log1p(delta / y) - delta * np.log(y + delta)
            - (1 - y) * np.log1p(-delta / (1 - y)) + delta * np.log(1 - y - delta)) / np.log(2)


def g_np(theta, a, b, c, d):
    """Same bound as physics_lim._fmc_log2_tail, numpy float64 (stable form)."""
    N = c + d
    s = d * theta / N
    Nxi = c * _dH_np(b, s) - d * _dH_np(b + s, c * theta / N)
    return 0.5 * np.log2(N / (c * d * b * (1 - b))) - Nxi - np.log2(a)


def theta_decimal(a, b, c, d, prec=60, iters=220):
    """Independent high-precision reference: the printed formula with naive xi, in 60-digit Decimal."""
    from decimal import Decimal as D, getcontext
    getcontext().prec = prec
    a, b, c, d = D(repr(a)), D(repr(b)), D(repr(c)), D(repr(d))
    ln2 = D(2).ln()
    H = lambda x: -(x * x.ln() + (1 - x) * (1 - x).ln()) / ln2
    N = c + d
    q = c / N
    g = lambda t: (N / (c * d * b * (1 - b))).ln() / (2 * ln2) - N * (H(b + t - q * t) - q * H(b) - (1 - q) * H(b + t)) - a.ln() / ln2
    lo, hi = D(0), (1 - b) * (1 - D("1e-12"))
    if g(hi) > 0:
        return float(hi)
    if g(D("1e-40")) <= 0:
        return 0.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        if g(mid) > 0:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2)


class TestExactTheta(unittest.TestCase):
    def test_against_brentq(self):
        n = 0
        for a, b, c, d in itertools.product([1e-17, 1e-12, 1e-6], [0.005, 0.03, 0.1, 0.3],
                                            [1e2, 1e4, 1e6, 1e9], [1e2, 2e3, 1e4, 1e6]):
            tmax = (1 - b) * (1 - 1e-12)
            th = float(theta_exact(jnp.asarray(a), jnp.asarray(b), jnp.asarray(c), jnp.asarray(d)))
            if g_np(1e-300, a, b, c, d) <= 0:
                self.assertEqual(th, 0.0)
                continue
            if g_np(tmax, a, b, c, d) > 0:
                self.assertAlmostEqual(th, tmax, delta=1e-12)
                continue
            ref = brentq(lambda t: g_np(t, a, b, c, d), 1e-300, tmax, xtol=1e-15, rtol=1e-14, maxiter=500)
            self.assertLess(abs(th - ref), 1e-12 + 1e-10 * ref, (a, b, c, d, th, ref))
            n += 1
        self.assertGreater(n, 100)

    def test_against_decimal_reference(self):
        """60-digit Decimal evaluation of the formula exactly as printed (naive xi) on a subset incl. c >> d."""
        for a, b, c, d in ((1e-12, 0.02, 1e9, 1e3), (1e-17, 0.03, 1e9, 1e2), (1e-10, 0.1, 1e5, 2e3),
                           (1e-12, 0.05, 1e4, 1e4), (1e-15, 0.2, 1e3, 1e8)):
            th = float(theta_exact(jnp.asarray(a), jnp.asarray(b), jnp.asarray(c), jnp.asarray(d)))
            ref = theta_decimal(a, b, c, d)
            self.assertLess(abs(th - ref), 1e-13 + 1e-11 * ref, (a, b, c, d, th, ref))

    def test_c_much_larger_than_d(self):
        for c, d in ((1e9, 1e3), (1e8, 2e3), (1e7, 50.0)):
            a, b = 1e-12, 0.02
            th = float(theta_exact(jnp.asarray(a), jnp.asarray(b), jnp.asarray(c), jnp.asarray(d)))
            ref = brentq(lambda t: g_np(t, a, b, c, d), 1e-300, (1 - b) * (1 - 1e-12), xtol=1e-15, rtol=1e-14)
            self.assertLess(abs(th - ref), 1e-12 + 1e-10 * ref)

    def test_check_grads_second_order(self):
        # in log coordinates for c, d (they span many decades)
        f = lambda la, b, lc, ld: theta_exact(jnp.exp(la), b, jnp.exp(lc), jnp.exp(ld))
        for la, b, lc, ld in ((np.log(1e-12), 0.05, np.log(1e5), np.log(2e3)),
                              (np.log(1e-10), 0.1, np.log(1e8), np.log(1e4)),
                              (np.log(1e-15), 0.02, np.log(3e6), np.log(3e6))):
            check_grads(f, tuple(jnp.asarray(x) for x in (la, b, lc, ld)), order=2, modes=["fwd", "rev"],
                        eps=1e-5, rtol=1e-5)

    def test_key_rate_grads_lim_tight(self):
        for p, L, nx in (([0.20, 0.10, 0.57, 0.30, 0.90], 20.0, 1e7), ([0.48, 0.35, 0.20, 0.75, 0.875], 50.0, 1e9)):
            f = lambda q: evaluate(q, L, nx, LIM_TIGHT.consts, LIM_TIGHT.flags)["R"] * 1e3
            check_grads(f, (jnp.asarray(p),), order=2, modes=["fwd", "rev"], eps=1e-6, rtol=2e-4)

    def test_max_mode(self):
        p, L, nx = jnp.asarray([0.20, 0.10, 0.57, 0.30, 0.90]), 20.0, 1e7
        lp = float(evaluate(p, L, nx, LIM.consts, LIM.flags)["l"])
        le = float(evaluate(p, L, nx, LIM_TIGHT.consts, LIM_TIGHT.flags)["l"])
        mx = with_flags(LIM, "lim_max", gamma_mode="max")
        lm = float(evaluate(p, L, nx, mx.consts, mx.flags)["l"])
        self.assertAlmostEqual(lm, min(lp, le), delta=1e-9 * abs(lm))


class TestPrintedUnchanged(unittest.TestCase):
    """l under gamma_mode='printed' equals values recorded before gamma_mode existed (commit db3bc92)."""
    PTS = [([0.49, 0.37, 0.19, 0.77, 0.88], 0.0, 1e9), ([0.48, 0.35, 0.20, 0.75, 0.875], 50.0, 1e9),
           ([0.57, 0.15, 0.08, 0.54, 0.19], 100.0, 1e4), ([0.20, 0.10, 0.57, 0.30, 0.90], 20.0, 1e7),
           ([0.45, 0.2, 0.3, 0.4, 0.6], 120.0, 1e6)]
    OLD = {"lim": [248128698.88382456, 251590223.1046997, 608.8745886787742, 2703399.508270526, 219493.73454735702],
           "lim_literal": [-100732271.24580283, -140183535.70585242, -2930.6538335736527, -2462966.2348506786,
                           -155280.09427054843],
           "legacy": [379584061.481341, 382954796.1258329, 2223.7267523893292, -945381.5384835467,
                      271171.5761472615]}

    def test_printed_equals_old(self):
        for pr in (LIM, LIM_LITERAL, LEGACY):
            self.assertEqual(pr.flags.gamma_mode, "printed")
            for (p, L, nx), old in zip(self.PTS, self.OLD[pr.name]):
                new = float(evaluate(jnp.asarray(p), L, nx, pr.consts, pr.flags)["l"])
                self.assertLessEqual(abs(new - old), 1e-12 * abs(old), (pr.name, L, nx, new, old))


if __name__ == "__main__":
    unittest.main()
