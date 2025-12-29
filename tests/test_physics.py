import unittest
import jax.numpy as jnp
from src.qkd.physics import calculate_eta_ch, calculate_eta_sys, calculate_h

class TestQKDPhysics(unittest.TestCase):

    def test_calculate_eta_ch(self):
        """Test channel transmittance calculation."""
        L = 100.0
        alpha = 0.2
        expected_eta = 10 ** (-alpha * L / 10)
        calculated_eta = calculate_eta_ch(L, alpha)
        self.assertAlmostEqual(calculated_eta, expected_eta, places=7)

    def test_calculate_eta_sys(self):
        """Test system transmittance calculation."""
        eta_Bob = 0.5
        eta_ch = 0.1
        expected = 0.05
        result = calculate_eta_sys(eta_Bob, eta_ch)
        self.assertAlmostEqual(result, expected, places=7)

    def test_calculate_h_binary_entropy(self):
        """Test binary entropy function."""
        # h(0.5) should be 1.0
        self.assertAlmostEqual(calculate_h(0.5), 1.0, places=5)
        # h(0) should be 0 (handled via clipping in the function)
        self.assertAlmostEqual(calculate_h(0.0), 0.0, places=4)
        
if __name__ == '__main__':
    unittest.main()
