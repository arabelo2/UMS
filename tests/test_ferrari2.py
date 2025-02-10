# tests/test_ferrari2.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
import warnings
import argparse
from domain.ferrari2 import ferrari2
from domain.interface2 import interface2

class TestFerrari2Complex(unittest.TestCase):
    def setUp(self):
        self.tol = 1e-6  # Tolerance for numerical comparisons

    def test_identical_media(self):
        """
        When the two media are identical (cr ~ 1), the explicit solution should be used:
            xi = DX * DT / (DF + DT)
        """
        cr = 1.0
        DF = 40.0   # Depth in medium two (mm)
        DT = 30.0   # Height in medium one (mm)
        DX = 50.0   # Separation distance (mm)
        expected = DX * DT / (DF + DT)  # 50*30/(40+30) ≈ 21.4286
        xi = ferrari2(cr, DF, DT, DX)
        self.assertAlmostEqual(xi, expected, places=6)

    def test_positive_DX(self):
        """
        For cr > 1 and positive DX, the computed xi should lie in [0, DX].
        """
        cr = 1.5
        DF = 40.0
        DT = 30.0
        DX = 50.0
        xi = ferrari2(cr, DF, DT, DX)
        self.assertGreaterEqual(xi, 0)
        self.assertLessEqual(xi, DX)

    def test_negative_DX(self):
        """
        For cr > 1 and negative DX, the computed xi should lie in [DX, 0].
        """
        cr = 1.5
        DF = 40.0
        DT = 30.0
        DX = -50.0
        xi = ferrari2(cr, DF, DT, DX)
        self.assertGreaterEqual(xi, DX)
        self.assertLessEqual(xi, 0)

    def test_fallback_branch_solution(self):
        """
        Use extreme parameters to force the fallback branch.
        Then verify that the final xi (in mm) makes interface2 nearly zero.
        """
        cr = 2.0      # Significantly different from 1 to force complex candidate roots.
        DF = 100.0
        DT = 10.0
        DX = 50.0
        xi = ferrari2(cr, DF, DT, DX)
        # In our updated code, xi is in mm directly. Therefore, call interface2 with xi.
        y_val = interface2(xi, cr, DF, DT, DX)
        self.assertAlmostEqual(y_val, 0.0, delta=self.tol)
        self.assertGreaterEqual(xi, 0)
        self.assertLessEqual(xi, DX)

    def test_solution_satisfies_interface2(self):
        """
        For a range of parameters, verify that the returned xi (in mm) yields a value 
        near zero when passed to interface2.
        """
        test_cases = [
            # Each tuple is (cr, DF, DT, DX)
            (1.5, 40.0, 30.0, 50.0),
            (2.0, 100.0, 10.0, 50.0),
            (1.2, 30.0, 20.0, 40.0),
            (1.8, 80.0, 15.0, 60.0),
        ]
        for (cr, DF, DT, DX) in test_cases:
            xi = ferrari2(cr, DF, DT, DX)
            # Now, call interface2 with xi (in mm) as input.
            y_val = interface2(xi, cr, DF, DT, DX)
            self.assertAlmostEqual(y_val, 0.0, delta=self.tol)

    def test_edge_case_tolerance(self):
        """
        Test parameters near the tolerance threshold for media identity.
        """
        cr = 1.0 + 1e-7  # Slightly above 1
        DF = 40.0
        DT = 30.0
        DX = 50.0
        expected = DX * DT / (DF + DT)
        xi = ferrari2(cr, DF, DT, DX)
        self.assertAlmostEqual(xi, expected, places=6)

    def test_invalid_input(self):
        """
        Test that non-numeric input for any parameter raises an error.
        """
        with self.assertRaises(Exception):
            ferrari2("not a number", 40.0, 30.0, 50.0)

if __name__ == '__main__':
    # Optionally, suppress runtime warnings (e.g., from invalid values in sqrt).
    import warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    unittest.main()
