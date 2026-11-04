"""Physics regression tests for src/qkd_opt (both presets). Run: python -m unittest discover tests"""
import os
import re
import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax.test_util import check_grads

from src.qkd_opt import LEGACY, LIM_EXACT, evaluate, with_flags

jax.config.update("jax_enable_x64", True)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# interior points (all clamps in src/qkd inactive), params = [mu1, mu2, P_mu1, P_mu2, q_x]
INTERIOR = [
    (np.array([0.49, 0.37, 0.19, 0.77, 0.88]), 0.0, 1e9),
    (np.array([0.48, 0.35, 0.20, 0.75, 0.875]), 50.0, 1e9),
    (np.array([0.57, 0.15, 0.08, 0.54, 0.19]), 100.0, 1e4),
    (np.array([0.20, 0.10, 0.57, 0.30, 0.90]), 20.0, 1e7),
]


class TestLimImplementation(unittest.TestCase):
    def test_matches_src_qkd_with_legacy_flags(self):
        """physics_lim with the legacy switches reproduces src/qkd (validates the transcription)."""
        lim_as_legacy = with_flags(LEGACY, "lim_as_legacy", model="lim")
        rng = np.random.default_rng(1)
        n = 0
        for L in (0.0, 50.0, 100.0, 140.0):
            for nx in (1e4, 1e6, 1e9):
                for _ in range(10):
                    p = np.array([rng.uniform(.3, .8), rng.uniform(.05, .25), rng.uniform(.05, .6),
                                  rng.uniform(.1, .35), rng.uniform(.2, .95)])
                    a = float(evaluate(p, L, nx, LEGACY.consts, LEGACY.flags)["l"])
                    b = float(evaluate(p, L, nx, lim_as_legacy.consts, lim_as_legacy.flags)["l"])
                    if a > 0:
                        n += 1
                        self.assertLess(abs(a - b) / abs(a), 1e-8, (p, L, nx, a, b))
        self.assertGreater(n, 20)

    def test_eps_fixed_point(self):
        c, f = LIM_EXACT.consts, LIM_EXACT.flags
        for p, L, nx in INTERIOR:
            out = evaluate(p, L, nx, c, f)
            l = float(out["l"])
            eps = float(out["eps_sec"])
            self.assertAlmostEqual(eps, c.kappa * max(l, 1.0), delta=1e-12 * eps)
            self.assertLess(float(out["eps_residual"]), 1e-12)

    def test_gradients_finite_and_correct(self):
        """jax.test_util.check_grads (fwd+rev, 2nd order) on R at several interior points."""
        for preset in (LEGACY, LIM_EXACT):
            for p, L, nx in INTERIOR:
                f = lambda q: evaluate(q, L, nx, preset.consts, preset.flags)["R"] * 1e3
                g = jax.grad(f)(jnp.asarray(p))
                self.assertTrue(np.all(np.isfinite(np.asarray(g))), (preset.name, L, nx))
                check_grads(f, (jnp.asarray(p),), order=2, modes=["fwd", "rev"], eps=1e-6, rtol=2e-4)

    def test_unclipped_beyond_horizon(self):
        """l and R are returned unclipped (negative) far beyond the horizon, with finite gradient."""
        for preset in (LEGACY, LIM_EXACT):
            p = jnp.asarray([0.48, 0.35, 0.20, 0.75, 0.875])
            out = evaluate(p, 200.0, 1e4, preset.consts, preset.flags)
            self.assertLess(float(out["l"]), 0.0)
            self.assertLess(float(out["R"]), 0.0)
            g = jax.grad(lambda q: evaluate(q, 200.0, 1e4, preset.consts, preset.flags)["l"])(p)
            self.assertTrue(np.all(np.isfinite(np.asarray(g))))
            self.assertGreater(float(jnp.max(jnp.abs(g))), 0.0)


class TestOptimisedPhysics(unittest.TestCase):
    """Uses the optimiser (cold, moderate effort) to test properties of the optimum."""

    @classmethod
    def setUpClass(cls):
        from src.qkd_opt.optimize import optimize_batch
        cls.optimize_batch = staticmethod(optimize_batch)
        L = np.array([0.0, 25.0, 50.0, 75.0, 100.0, 125.0])
        cls.L = L
        cls.res = {pr.name: optimize_batch(L, np.full_like(L, 1e9), pr, n_sobol=2048, top_k=6, seed=3)
                   for pr in (LEGACY, LIM_EXACT)}

    def test_monotone_decay_in_L(self):
        for name, r in self.res.items():
            R = r["R"]
            self.assertTrue(np.all(np.diff(R) < 0), (name, R))

    def test_legacy_value_at_L0(self):
        """Legacy optimum at L=0, n_X=1e9 (independent of the old runs, which all report 1.0900e-2)."""
        R0 = self.res["legacy"]["R"][0]
        self.assertAlmostEqual(R0 / 1.0899651e-2, 1.0, delta=1e-6)

    def test_lim_exact_value_at_L0(self):
        """lim_exact at L=0, n_X=1e9: regression value 1.2504692e-3 (your stated Lim value ~1.25e-3)."""
        R0 = self.res["lim_exact"]["R"][0]
        self.assertAlmostEqual(R0 / 1.2504692e-3, 1.0, delta=1e-5)

    def test_certificate_present(self):
        for r in self.res.values():
            for k in ("grad_norm_theta", "hess_eig", "active_bounds", "status"):
                self.assertIn(k, r)
            self.assertTrue(np.all(r["hess_eig"].min(axis=1) > -1e-6))   # local maxima of R


class TestLegacyTailRegression(unittest.TestCase):
    """Under the legacy preset the new optimiser must not lose to the stored Hessian (FoR) values in
    assets/tail_region_analysis_140_150km.md (n_X = 1e4, 140-150 km)."""

    def test_new_optimizer_beats_stored_hessian_tail(self):
        """Production path (warm continuation sweep on the legacy 400-point grid from 120 km).
        Note: a cold start with the default budget (2^12 Sobol, top-8) does NOT find the narrow
        basin at 147.37 km (1.812e-10 vs 2.0453e-10); only 2^17/top-64 does. This is reported in
        the benchmark (cold vs warm) rather than hidden here."""
        path = os.path.join(ROOT, "assets", "tail_region_analysis_140_150km.md")
        if not os.path.exists(path):
            self.skipTest("assets/tail_region_analysis_140_150km.md not present")
        with open(path) as fh:
            text = fh.read().split("## Pure Dual Annealing")[0]
        rows = re.findall(r"^\|([0-9.]+)\|([0-9.eE+-]+)\|", text, flags=re.M)
        L = np.array([float(a) for a, _ in rows])
        R_stored = np.array([float(b) for _, b in rows])
        self.assertEqual(len(L), 15)
        from src.qkd_opt.optimize import optimize_sweep
        grid = np.linspace(0, 200, 400)
        sub = grid[(grid >= 120) & (grid <= 150)]
        s = optimize_sweep(sub, [1e4], LEGACY, global_every=10, n_sobol_init=16384, top_k_init=16,
                           refine_horizon=False)
        for Li, Rs in zip(L, R_stored):
            j = int(np.argmin(np.abs(sub - Li)))
            self.assertLess(abs(sub[j] - Li), 1e-5)
            Rn = s["R"][0, j]
            # stored values are printed with 5 significant digits
            self.assertGreaterEqual(Rn, Rs * (1 - 5e-5), f"L={Li}: new {Rn:.6e} < stored {Rs:.6e}")

if __name__ == "__main__":
    unittest.main()
