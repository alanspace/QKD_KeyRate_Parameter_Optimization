"""qml study, main-.venv tests (unconditional; PREREGISTRATION.md sections 1, 4, 6).

- shared constants in qml/common.py equal the ones in the code they mirror
- every model output z in [-1, 1]^5 maps through the fixed affine map and reparam to a feasible p
- certification at the reference parameters reproduces the reference R within tol_ref
- splits of the committed label file are disjoint and follow the lattice rules
"""
import glob
import json
import os
import unittest

import numpy as np

from qml import common as C

ROOT = C.ROOT


class TestConstants(unittest.TestCase):
    def test_mirrors(self):
        from src.qkd_opt import reparam
        self.assertEqual(C.U_LO, reparam.U_LO)
        from Optimization.reference import build_reference as BR
        self.assertEqual((C.CLIFF_GTOL, C.NOISE_GTOL), (BR.CLIFF_GTOL, BR.NOISE_GTOL))


class TestFeasibility(unittest.TestCase):
    def test_any_z_is_feasible(self):
        import jax
        jax.config.update("jax_enable_x64", True)
        from src.qkd_opt import reparam
        rng = np.random.default_rng(0)
        u_train = rng.uniform(0.02, 0.95, (500, 5))
        lo, hi = C.u_range(u_train)
        z = np.concatenate([rng.uniform(-1, 1, (20000, 5)), np.full((1, 5), -1.0), np.full((1, 5), 1.0),
                            rng.uniform(-50, 50, (2000, 5))])            # out-of-range z is clipped
        u = C.z_to_u(z, lo, hi)
        self.assertTrue(np.all((u > 0) & (u < 1)))
        p = np.asarray(reparam.params_of_u(u, 2e-4))
        mu1, mu2, P1, P2, qx = p.T
        mu3 = 2e-4
        self.assertTrue(np.all(mu2 > mu3))
        self.assertTrue(np.all(mu1 > mu2 + mu3))
        self.assertTrue(np.all(mu1 < reparam.MU1_MAX) and np.all(mu2 < reparam.MU2_MAX))
        self.assertTrue(np.all((P1 > 0) & (P2 > 0) & (1 - P1 - P2 > 0)))
        self.assertTrue(np.all((qx > 0) & (qx < 1)))


class TestCertifyReproducesReference(unittest.TestCase):
    def test_reference_points(self):
        from qml import certify as CE
        from src.qkd_opt import reparam
        for tag in ("", "_tail"):
            ref = json.load(open(os.path.join(ROOT, "Optimization", "reference", f"reference_lim{tag}.json")))
            tol = ref["summary"]["tol_ref"]
            pts = [p for p in ref["points"] if p["R"] > 0]
            P = np.array([p["params"] for p in pts])
            u = reparam.u_of_params(P, CE.MU3)
            R, _ = CE.R_at(CE.theta_of(u), np.array([p["L"] for p in pts]), np.array([p["n_X"] for p in pts]))
            Rr = np.array([p["R"] for p in pts])
            rel = np.abs(R - Rr) / Rr
            self.assertLessEqual(float(rel.max()), tol, f"reference_lim{tag}: max rel {rel.max():.3e} > tol_ref {tol}")


class TestSplits(unittest.TestCase):
    def setUp(self):
        files = sorted(glob.glob(os.path.join(C.DATA, "labels_*.npz")))
        if not files:
            self.skipTest("no committed label file in qml/data (run qml/labels.py)")
        self.d = dict(np.load(files[-1], allow_pickle=False))

    def test_disjoint_and_lattice_rules(self):
        d = self.d
        key = lambda m: set(zip(d["n_X"][m].tolist(), d["L"][m].tolist()))
        train = key(d["split"] == "train")
        for s in ("val", "test", "ref", "heldout", "extrap"):
            self.assertFalse(train & key(d["split"] == s), f"{s} overlaps train")
        k = d["lattice_k"]
        tr = d["split"] == "train"
        bulk, tail = d["region"] == "bulk", d["region"] == "tail"
        self.assertTrue(np.all(k[tr & bulk] % 16 == 0))
        self.assertTrue(np.all(k[tr & tail] % 4 == 0))
        self.assertTrue(np.all(np.isin(k[(d["split"] == "test") & bulk] % 16, [8, 12])))
        self.assertTrue(np.all(np.isin(k[(d["split"] == "test") & tail] % 4, [2, 3])))
        heldout = np.isin(d["n_X"], C.HELDOUT_NX + C.EXTRAP_NX)
        self.assertFalse(np.any(heldout & np.isin(d["split"], ["train", "val"])))
        self.assertFalse(np.any(d["cliff_edge"] & d["scored"] & (d["split"] != "ref")))


if __name__ == "__main__":
    unittest.main()
