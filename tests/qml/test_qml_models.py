""".venv-qml tests: skipped, with the reason printed, when PennyLane is not installed (e.g. CI, main .venv)."""
import unittest

try:
    import pennylane  # noqa: F401
    HAVE_PL = True
except ImportError:
    HAVE_PL = False
    print("tests/qml/test_qml_models.py: skipped, PennyLane not installed (run in .venv-qml)")


@unittest.skipUnless(HAVE_PL, "PennyLane not installed: .venv-qml tests skipped")
class TestModels(unittest.TestCase):
    def test_parameter_counts_and_range(self):
        import jax.numpy as jnp
        import numpy as np
        from qml import models as M
        X = M.xprime(jnp.linspace(0, 1.05, 7), jnp.linspace(-0.1, 1.1, 7))
        for (n, L), h in zip(M.PQC_GRID, M.NN_WIDTHS):
            for ent in (True, False):
                init, apply, npar = M.make_pqc(n, L, ent)
                w = init(0)
                self.assertEqual(M.n_params(w), npar)
                z = np.asarray(apply(w, X))
                self.assertEqual(z.shape, (7, 5))
                self.assertTrue(np.all(np.abs(z) <= 1 + 1e-12))
            init, apply, nn = M.make_nn(h)
            self.assertEqual(M.n_params(init(0)), nn)
            self.assertLessEqual(abs(nn - 5 * n * L) / (5 * n * L), 0.10, f"NN h={h} not matched to PQC {n}x{L}")

    def test_no_physics_import(self):
        import sys
        from qml import models, pilot_train  # noqa: F401
        self.assertFalse(any(m.startswith("src.qkd") for m in sys.modules), "physics imported in .venv-qml code")


if __name__ == "__main__":
    unittest.main()
