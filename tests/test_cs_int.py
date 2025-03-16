# tests/test_cs_int.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import unittest
import numpy as np
from domain.cs_int import cs_int

class TestCsIntRobust(unittest.TestCase):

    def test_scalar_zero(self):
        # When xi = 0, we expect the approximations to yield:
        # c = 0 and s = 0, based on the formulas.
        xi = 0
        c, s = cs_int(xi)
        self.assertAlmostEqual(c, 0.0, places=6)
        self.assertAlmostEqual(s, 0.0, places=6)

    def test_scalar_small_positive(self):
        # Test with a very small positive number.
        xi = 1e-6
        c, s = cs_int(xi)
        # Check that the result is finite.
        self.assertTrue(np.isfinite(c))
        self.assertTrue(np.isfinite(s))

    def test_scalar_positive(self):
        # Test with a typical positive scalar.
        xi = 1.0
        c, s = cs_int(xi)
        # Check that the outputs are finite.
        self.assertTrue(np.isfinite(c))
        self.assertTrue(np.isfinite(s))
        # For rough comparison, the expected values (approximate) for xi=1 are:
        # c ~ 0.78, s ~ 0.44. Allow a relative tolerance of 1e-2.
        np.testing.assert_allclose(c, 0.78, rtol=1e-2)
        np.testing.assert_allclose(s, 0.44, rtol=1e-2)

    def test_vector_input(self):
        # Test with a vector input.
        xi = np.linspace(0, 2, 50)
        c, s = cs_int(xi)
        # The output arrays should have the same shape as the input.
        self.assertEqual(c.shape, xi.shape)
        self.assertEqual(s.shape, xi.shape)
        # All values should be finite.
        self.assertTrue(np.all(np.isfinite(c)))
        self.assertTrue(np.all(np.isfinite(s)))

    def test_empty_input(self):
        # Test with empty input. We expect an empty array as output.
        xi = []
        c, s = cs_int(xi)
        self.assertEqual(np.size(c), 0)
        self.assertEqual(np.size(s), 0)

    def test_non_numeric_input(self):
        # Test that non-numeric input raises an error.
        with self.assertRaises(ValueError):
            cs_int("non-numeric")

    def test_negative_input(self):
        # Even though the approximation is intended for positive values,
        # we test that negative inputs do not crash the function and produce finite output.
        xi = -np.linspace(0, 2, 50)  # negative values
        c, s = cs_int(xi)
        self.assertTrue(np.all(np.isfinite(c)))
        self.assertTrue(np.all(np.isfinite(s)))
        # Optionally, we could print a warning if negative values are used, but for now we simply check for finite outputs.

if __name__ == '__main__':
    unittest.main()
