# tests/test_ps_3Dv.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
from application.ps_3Dv_service import run_ps_3Dv_service

class TestPS3DvService(unittest.TestCase):
    def test_1D_simulation(self):
        # 1D: x and y as scalars, z as a vector.
        lx = 6
        ly = 12
        f = 5
        c = 1480
        ex = 0
        ey = 0
        x = 0
        y = 0
        z = np.linspace(5, 100, 400)
        p = run_ps_3Dv_service(lx, ly, f, c, ex, ey, x, y, z)
        self.assertEqual(p.shape, z.shape)
        self.assertTrue(np.iscomplexobj(p))

    def test_2D_simulation(self):
        # 2D: x and y as arrays (meshgrid simulation).
        lx = 6
        ly = 12
        f = 5
        c = 1480
        ex = 0
        ey = 0
        x = np.linspace(-5, 5, 50)
        y = np.linspace(-5, 5, 50)
        z = 50  # Scalar z.
        # Create meshgrid for a full 2D evaluation.
        X, Y = np.meshgrid(x, y)
        p = run_ps_3Dv_service(lx, ly, f, c, ex, ey, X, Y, z)
        self.assertEqual(p.shape, X.shape)
        self.assertTrue(np.iscomplexobj(p))

    def test_optional_integration_params(self):
        # Test that specifying P and Q returns reasonable results.
        lx = 6
        ly = 12
        f = 5
        c = 1480
        ex = 0
        ey = 0
        x = 0
        y = 0
        z = np.linspace(5, 100, 400)
        p_auto = run_ps_3Dv_service(lx, ly, f, c, ex, ey, x, y, z)
        p_manual = run_ps_3Dv_service(lx, ly, f, c, ex, ey, x, y, z, P=50, Q=50)
        # They need not be identical, but should be within a reasonable tolerance.
        np.testing.assert_allclose(p_auto, p_manual, rtol=0.2)

if __name__ == '__main__':
    unittest.main()
