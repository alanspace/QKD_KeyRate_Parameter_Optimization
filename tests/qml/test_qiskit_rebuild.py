"""Addition C: the Qiskit rebuild reproduces the PennyLane expectations to <= 1e-10 (.venv-qml; skipped elsewhere).
Uses the trained dense seed-0 weights when present, and random weights in any case."""
import unittest

try:
    import pennylane  # noqa: F401
    import qiskit  # noqa: F401
    HAVE = True
except ImportError:
    HAVE = False
    print("tests/qml/test_qiskit_rebuild.py: skipped, PennyLane/Qiskit not installed (run in .venv-qml)")


@unittest.skipUnless(HAVE, "PennyLane/Qiskit not installed: .venv-qml tests skipped")
class TestRebuild(unittest.TestCase):
    def test_rebuild_matches(self):
        import types
        from qml.iqm import rebuild as RB
        ok, rows = RB.cmd_check(types.SimpleNamespace(random=True))
        self.assertTrue(ok, rows)


if __name__ == "__main__":
    unittest.main()
