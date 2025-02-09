"""
Module: test_ls_2Dint.py
Layer: Tests

Provides unit tests for the LS2DInterfaceService.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
from application.ls_2Dint_service import LS2DInterfaceService

class TestLS2Dint(unittest.TestCase):
    def setUp(self):
        # Example parameters for testing.
        self.b = 3.0          # mm
        self.f = 5.0          # MHz
        self.mat = [1, 1480, 7.9, 5900]  # d1, c1, d2, c2
        self.e = 0.0          # mm
        self.angt = 10.217    # degrees
        self.Dt0 = 50.8       # mm
        # For testing the 2D case, create a default meshgrid.
        self.x = np.linspace(0, 25, 200)
        self.z = np.linspace(1, 25, 200)
        self.xx, self.zz = np.meshgrid(self.x, self.z)
        self.service = LS2DInterfaceService(self.b, self.f, self.mat, self.e, self.angt, self.Dt0, self.xx, self.zz)

    def test_pressure_type(self):
        """Test that the computed pressure is either a complex scalar or an array with complex dtype."""
        p = self.service.calculate()
        if np.isscalar(p):
            self.assertIsInstance(p, complex)
        else:
            # p should be a numpy array and its dtype should be a subdtype of complex.
            self.assertTrue(hasattr(p, "dtype"))
            self.assertTrue(np.issubdtype(p.dtype, np.complexfloating))

    def test_pressure_nonzero(self):
        """Test that the computed pressure is non-zero (for these test parameters)."""
        p = self.service.calculate()
        if np.isscalar(p):
            self.assertNotEqual(abs(p), 0)
        else:
            # Ensure that at least one element of the array is non-zero.
            self.assertTrue(np.any(np.abs(p) != 0), "Pressure array is entirely zero.")

if __name__ == "__main__":
    unittest.main()
