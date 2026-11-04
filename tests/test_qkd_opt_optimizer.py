import unittest

import jax
import numpy as np

from src.qkd_opt import LEGACY
from src.qkd_opt.optimize import optimize_batch, optimize_sweep, global_stage

jax.config.update("jax_enable_x64", True)


class TestOptimizer(unittest.TestCase):
    def test_local_beats_dense_sobol(self):
        """Refined optimum is at least as good as the best of 2^15 Sobol points (no refinement)."""
        L, nx = np.array([60.0, 150.0]), np.array([1e6, 1e4])
        _, R_sobol, _ = global_stage(L, nx, LEGACY, n_sobol=2**15, top_k=1, seed=11)
        r = optimize_batch(L, nx, LEGACY, n_sobol=1024, top_k=4, seed=0)
        self.assertTrue(np.all(r["R"] >= R_sobol[:, 0]))

    def test_beyond_horizon_flagged_not_clipped(self):
        r = optimize_batch([200.0], [1e4], LEGACY, n_sobol=1024, top_k=4)
        self.assertIn("beyond_horizon", r["status"][0])
        self.assertLess(r["l"][0], 0)
        self.assertEqual(r["objective_mode"][0], "l/n_X")
        self.assertTrue(np.all(np.isfinite(r["params"][0])))

    def test_sweep_branch_switch_logged(self):
        """n_X=1e9 legacy has a branch crossover near 154 km (REVIEW.md 2b); the sweep must log it."""
        Lg = np.arange(130.0, 171.0, 2.0)
        s = optimize_sweep(Lg, [1e9], LEGACY, global_every=5, n_sobol_init=4096, top_k_init=8,
                           refine_horizon=False)
        sw = [e for e in s["events"] if e["type"] == "branch_switch" and e["objective"] == "R"]
        self.assertGreaterEqual(len(sw), 1)
        bf = [e for e in s["events"] if e["type"] == "backfill"]
        lo, hi = bf[0]["crossover_between_km"]
        self.assertTrue(146 <= lo <= 160 and 146 <= hi <= 162, (lo, hi))


if __name__ == "__main__":
    unittest.main()
