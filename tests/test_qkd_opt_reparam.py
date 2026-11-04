import unittest

import jax
import jax.numpy as jnp
import numpy as np

from src.qkd_opt import reparam

jax.config.update("jax_enable_x64", True)
MU3 = 2e-4


class TestReparam(unittest.TestCase):
    def test_constraints_hold_by_construction(self):
        rng = np.random.default_rng(0)
        th = np.concatenate([rng.normal(0, 3, (5000, 5)), rng.normal(0, 60, (5000, 5)),
                             np.array([[50.0] * 5, [-50.0] * 5, [800.0, -800, 800, -800, 800]])])
        p = np.asarray(reparam.params_of_theta(jnp.asarray(th), MU3))
        mu1, mu2, P1, P2, qx = p.T
        P3 = 1 - P1 - P2
        self.assertTrue(np.all(mu1 > mu2 + MU3))
        self.assertTrue(np.all(mu2 > MU3))
        self.assertTrue(np.all(mu1 <= reparam.MU1_MAX) and np.all(mu2 <= reparam.MU2_MAX))
        for x in (P1, P2, P3, qx, 1 - qx):
            self.assertTrue(np.all(x > 0))
        self.assertTrue(np.all(np.isfinite(p)))

    def test_round_trip(self):
        rng = np.random.default_rng(1)
        th = rng.normal(0, 2, (1000, 5))
        p = np.asarray(reparam.params_of_theta(jnp.asarray(th), MU3))
        th2 = reparam.theta_of_params(p, MU3)
        np.testing.assert_allclose(th2, th, rtol=1e-6, atol=1e-6)

    def test_active_bounds(self):
        self.assertEqual(reparam.active_bounds([0.5] * 5), [])
        self.assertIn("q_x->0", reparam.active_bounds([0.5, 0.5, 0.5, 0.5, 1e-9]))


if __name__ == "__main__":
    unittest.main()
