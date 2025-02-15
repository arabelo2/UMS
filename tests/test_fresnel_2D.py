# tests/test_fresnel_2D.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
from application.fresnel_2D_service import run_fresnel_2D_service

class TestFresnel2DService(unittest.TestCase):

    def test_scalar_inputs(self):
        """Test with scalar x and scalar z."""
        b = 6
        f = 5
        c = 1500
        x = 0        # scalar x
        z = 60       # scalar z
        p = run_fresnel_2D_service(b, f, c, x, z)
        # p should be a scalar (or 0-dim array) and complex
        self.assertTrue(np.isscalar(p) or (isinstance(p, np.ndarray) and p.shape == ()))
        self.assertTrue(np.iscomplex(p))
        self.assertTrue(np.isfinite(np.real(p)))
        self.assertTrue(np.isfinite(np.imag(p)))

    def test_vector_x_scalar_z(self):
        """Test with vector x and scalar z."""
        b = 6
        f = 5
        c = 1500
        x = np.linspace(-10, 10, 200)
        z = 60  # scalar z
        p = run_fresnel_2D_service(b, f, c, x, z)
        self.assertEqual(p.shape, x.shape)
        self.assertTrue(np.all(np.isfinite(np.real(p))))
        self.assertTrue(np.all(np.isfinite(np.imag(p))))

    def test_scalar_x_vector_z(self):
        """Test with scalar x and vector z."""
        b = 6
        f = 5
        c = 1500
        x = 0  # scalar x
        z = np.linspace(1, 100, 200)
        p = run_fresnel_2D_service(b, f, c, x, z)
        self.assertEqual(p.shape, z.shape)
        self.assertTrue(np.all(np.isfinite(np.real(p))))
        self.assertTrue(np.all(np.isfinite(np.imag(p))))

    def test_vector_x_vector_z(self):
        """
        Test with both x and z as vectors.
        In this conversion, if both x and z are arrays of the same shape,
        the function should perform element-wise computations.
        """
        b = 6
        f = 5
        c = 1500
        x = np.linspace(-10, 10, 200)
        # Use a vector for z that is compatible (same shape as x)
        z = np.linspace(1, 100, 200)
        p = run_fresnel_2D_service(b, f, c, x, z)
        self.assertEqual(p.shape, x.shape)
        self.assertTrue(np.all(np.isfinite(np.real(p))))
        self.assertTrue(np.all(np.isfinite(np.imag(p))))

    def test_zero_z_handling(self):
        """
        Test that if z is 0 (which would cause division by zero),
        the function uses a small epsilon internally and returns a finite result.
        """
        b = 6
        f = 5
        c = 1500
        x = 0
        z = 0   # Should trigger epsilon replacement
        p = run_fresnel_2D_service(b, f, c, x, z)
        self.assertTrue(np.isfinite(np.real(p)))
        self.assertTrue(np.isfinite(np.imag(p)))

    def test_invalid_input(self):
        """Test that non-numeric input for x raises an error."""
        b = 6
        f = 5
        c = 1500
        x = "invalid"
        z = 60
        with self.assertRaises(Exception):
            run_fresnel_2D_service(b, f, c, x, z)

if __name__ == '__main__':
    unittest.main()
