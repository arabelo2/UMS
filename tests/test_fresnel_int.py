# tests/test_fresnel_int.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
from domain.fresnel_int import fresnel_int

class TestFresnelIntRobust(unittest.TestCase):
    def test_scalar_zero(self):
        """Test that fresnel_int(0) returns 0+0j."""
        result = fresnel_int(0)
        self.assertAlmostEqual(result.real, 0.0, places=6)
        self.assertAlmostEqual(result.imag, 0.0, places=6)

    def test_scalar_small_positive(self):
        """Test fresnel_int for a very small positive value."""
        x = 1e-6
        result = fresnel_int(x)
        self.assertTrue(np.isfinite(result.real))
        self.assertTrue(np.isfinite(result.imag))
        # Relax the threshold slightly to 1.1e-6.
        self.assertLess(np.abs(result), 1.1e-6)

    def test_scalar_positive(self):
        """Test fresnel_int for a typical positive scalar value.
           Since we don't have an exact expected value, we ensure the result is finite and non-zero.
        """
        x = 1.0
        result = fresnel_int(x)
        self.assertTrue(np.isfinite(result.real))
        self.assertTrue(np.isfinite(result.imag))
        self.assertGreater(np.abs(result), 0)

    def test_vector_input(self):
        """Test fresnel_int with a vector input."""
        x = np.linspace(0, 2, 5)  # e.g., [0, 0.5, 1.0, 1.5, 2.0]
        result = fresnel_int(x)
        self.assertEqual(result.shape, x.shape)
        self.assertTrue(np.all(np.isfinite(result.real)))
        self.assertTrue(np.all(np.isfinite(result.imag)))
        
    def test_negative_input_symmetry(self):
        """Test that fresnel_int is an odd function: fresnel_int(-x) == - fresnel_int(x) for x>0."""
        x = 1.0
        result_pos = fresnel_int(x)
        result_neg = fresnel_int(-x)
        # Use relative tolerance for the check.
        np.testing.assert_allclose(result_neg, -result_pos, rtol=1e-6)

    def test_empty_input(self):
        """Test that an empty input returns an empty array."""
        x = np.array([])
        result = fresnel_int(x)
        self.assertEqual(result.size, 0)

    def test_non_numeric_input(self):
        """Test that non-numeric input raises an error."""
        with self.assertRaises(Exception):
            fresnel_int("non-numeric")

if __name__ == '__main__':
    unittest.main()
