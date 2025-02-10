# tests/test_interface2.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
import math
import warnings
from domain.interface2 import interface2

class TestInterface2EdgeCases(unittest.TestCase):
    def test_scalar_normal(self):
        """
        Test a normal scalar input where x is nonzero.
        """
        x = 10.0
        cr = 1.5
        df = 20.0
        dp = 30.0
        dpf = 40.0
        # Manually compute expected terms:
        term1 = x / np.sqrt(x**2 + dp**2)
        term2 = cr * (dpf - x) / np.sqrt((dpf - x)**2 + df**2)
        expected = term1 - term2
        y = interface2(x, cr, df, dp, dpf)
        self.assertAlmostEqual(y, expected, places=6)

    def test_scalar_x_zero_dp_nonzero(self):
        """
        Test the case where x = 0 and dp != 0.
        Expected:
          term1 = 0 / sqrt(0+dp^2) = 0,
          term2 = cr*(dpf)/sqrt(dpf^2+df^2),
          so y = -cr*(dpf)/sqrt(dpf^2+df^2)
        """
        x = 0.0
        cr = 1.2
        df = 25.0
        dp = 35.0
        dpf = 45.0
        expected = -cr * dpf / np.sqrt(dpf**2 + df**2)
        y = interface2(x, cr, df, dp, dpf)
        self.assertAlmostEqual(y, expected, places=6)

    def test_scalar_x_zero_dp_zero(self):
        """
        Test the case where x = 0 and dp = 0.
        This leads to a term 0/sqrt(0) in term1.
        We expect the result to be NaN (Not a Number) due to 0/0.
        """
        x = 0.0
        cr = 1.2
        df = 25.0
        dp = 0.0
        dpf = 45.0
        y = interface2(x, cr, df, dp, dpf)
        self.assertTrue(np.isnan(y), "Expected NaN when x=0 and dp=0 (division by zero)")

    def test_array_input_mixed_values(self):
        """
        Test with an array input that includes negative, zero, and positive values.
        Ensure that the output shape matches the input shape and values are computed correctly.
        """
        x = np.array([-5, 0, 5, 10])
        cr = 1.3
        df = 30.0
        dp = 40.0
        dpf = 50.0
        y = interface2(x, cr, df, dp, dpf)
        self.assertEqual(y.shape, x.shape)
        # Check each element against the manual calculation.
        for i, xi in enumerate(x):
            term1 = xi / np.sqrt(xi**2 + dp**2)
            term2 = cr * (dpf - xi) / np.sqrt((dpf - xi)**2 + df**2)
            expected = term1 - term2
            if np.isnan(expected):
                self.assertTrue(np.isnan(y[i]))
            else:
                self.assertAlmostEqual(y[i], expected, places=6)

    def test_large_values(self):
        """
        Test with very large x values relative to dp.
        For x = 1e6, dp = 30, dpf = 50, df = 20, and cr = 1.1:
          - term1 approximates to 1 (since sqrt(x^2 + dp^2) ~ x)
          - term2 approximates to 1.1 * (dpf - x)/abs(dpf - x) = -1.1  (since dpf - x is nearly -x)
          - Thus, y = 1 - (-1.1) = 2.1 approximately.
        """
        x = 1e6
        cr = 1.1
        df = 20.0
        dp = 30.0
        dpf = 50.0
        y = interface2(x, cr, df, dp, dpf)
        self.assertAlmostEqual(y, 2.1, delta=1e-6)

    def test_empty_array(self):
        """
        Test that an empty array input returns an empty array.
        """
        x = np.array([])
        cr = 1.0
        df = 20.0
        dp = 30.0
        dpf = 40.0
        y = interface2(x, cr, df, dp, dpf)
        self.assertEqual(y.size, 0)

    def test_invalid_type(self):
        """
        Test that passing a non-numeric input (e.g., a string) raises an error.
        """
        with self.assertRaises(Exception):
            interface2("not a number", 1.2, 20.0, 30.0, 40.0)

if __name__ == '__main__':
    # Optionally, suppress runtime warnings (e.g., division by zero warnings)
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    unittest.main()
