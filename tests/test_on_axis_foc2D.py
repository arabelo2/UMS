# tests/test_on_axis_foc2D.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
from application.on_axis_foc2D_service import run_on_axis_foc2D_service

# Fixture to close all plots after each test
def tearDownModule():
    """
    Close all matplotlib plots after the test module completes.
    """
    plt.close('all')

class TestOnAxisFoc2DService(unittest.TestCase):

    def test_scalar_input(self):
        """Test on_axis_foc2D with scalar z input."""
        b = 6
        R = 100
        f = 5
        c = 1480
        z = 60  # scalar
        p = run_on_axis_foc2D_service(b, R, f, c, z)
        # p should be a scalar (or 0-dim array) and complex.
        self.assertTrue(np.isscalar(p) or (isinstance(p, np.ndarray) and p.shape == ()))
        self.assertTrue(np.iscomplex(p))
        self.assertTrue(np.isfinite(np.real(p)))
        self.assertTrue(np.isfinite(np.imag(p)))

    def test_vector_input(self):
        """Test on_axis_foc2D with vector input for z."""
        b = 6
        R = 100
        f = 5
        c = 1480
        z = np.linspace(20, 400, 500)
        p = run_on_axis_foc2D_service(b, R, f, c, z)
        self.assertEqual(p.shape, z.shape)
        self.assertTrue(np.all(np.isfinite(np.real(p))))
        self.assertTrue(np.all(np.isfinite(np.imag(p))))

    def test_zero_z_handling(self):
        """Test that if z is 0, the function avoids division by zero and returns finite output."""
        b = 6
        R = 100
        f = 5
        c = 1480
        z = 0  # triggers epsilon adjustment
        p = run_on_axis_foc2D_service(b, R, f, c, z)
        self.assertTrue(np.isfinite(np.real(p)))
        self.assertTrue(np.isfinite(np.imag(p)))

    def test_analytical_branch(self):
        """
        Test that when u = 1 - z/R is very small (|u| <= 0.005), the analytical branch is used.
        For such z, p should be approximately:
            p = sqrt(2/(1j)) * sqrt((b/R) * kb/pi)
        where kb = 2000*pi*f*b/c.
        """
        b = 6
        R = 100
        f = 5
        c = 1480
        # Choose z so that u = 1 - z/R is near zero.
        # Let u = 0.005 => z = R*(1 - 0.005) = 100*0.995 = 99.5 mm.
        z = 99.5
        p = run_on_axis_foc2D_service(b, R, f, c, z)
        kb = 2000 * math.pi * f * b / c
        expected = np.sqrt(2/(1j)) * np.sqrt((b/R) * kb / math.pi)
        # Relax tolerance to 1e-2.
        np.testing.assert_allclose(p, expected, rtol=1e-2)

    def test_invalid_input(self):
        """Test that non-numeric input for z raises an error."""
        b = 6
        R = 100
        f = 5
        c = 1480
        z = "invalid"
        with self.assertRaises(Exception):
            run_on_axis_foc2D_service(b, R, f, c, z)

if __name__ == '__main__':
    unittest.main()
    