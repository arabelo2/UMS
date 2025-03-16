#!/usr/bin/env python3
# tests/test_mls_array_modeling.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import unittest
import numpy as np
import math

# Import modules from your project:
from domain.elements import ElementsCalculator
from domain.delay_laws2D import delay_laws2D
from application.mls_array_modeling_service import run_mls_array_modeling_service
from application.discrete_windows_service import run_discrete_windows_service
from application.ls_2Dv_service import run_ls_2Dv_service

class TestElementsCalculator(unittest.TestCase):
    def test_calculate(self):
        # Test with known parameters.
        f = 5       # MHz
        c = 1480    # m/s
        dl = 0.5
        gd = 0.1
        M = 32
        calc = ElementsCalculator(f, c, dl, gd, M)
        A, d, g, e = calc.calculate()
        # Check that A, d, and g are numbers and e is an array of length M.
        self.assertIsInstance(A, (int, float))
        self.assertIsInstance(d, (int, float))
        self.assertIsInstance(g, (int, float))
        self.assertIsInstance(e, np.ndarray)
        self.assertEqual(len(e), M)
        # Consistency check: overall aperture should equal M*d + (M-1)*g.
        expected_A = M * d + (M - 1) * g
        self.assertAlmostEqual(A, expected_A, places=5)

class TestDelayLaws2D(unittest.TestCase):
    def test_delay_output_shape(self):
        # Test that the delay function returns an array of correct length.
        M = 32
        s = 1.0  # mm (example pitch)
        Phi = 20.0
        F = np.inf
        c = 1480.0
        td = delay_laws2D(M, s, Phi, F, c)
        self.assertEqual(td.shape[0], M)
        self.assertTrue(np.all(np.isfinite(td)))
        
    def test_delay_different_angles(self):
        # Test that changing Phi yields different delays.
        M = 32
        s = 1.0
        F = np.inf
        c = 1480.0
        td1 = delay_laws2D(M, s, 20.0, F, c)
        td2 = delay_laws2D(M, s, -20.0, F, c)
        # They should not be identical.
        self.assertFalse(np.allclose(td1, td2))

class TestDiscreteWindows(unittest.TestCase):
    def test_window_rect(self):
        M = 32
        wtype = 'rect'
        Ct = run_discrete_windows_service(M, wtype)
        self.assertEqual(len(Ct), M)
        # For rectangular window, we expect all ones.
        self.assertTrue(np.allclose(Ct, np.ones(M, dtype=complex)))
    
    def test_window_cos(self):
        M = 32
        wtype = 'cos'
        Ct = run_discrete_windows_service(M, wtype)
        self.assertEqual(len(Ct), M)
        # Check that values are in a plausible range (0 to 1).
        self.assertTrue(np.all((Ct >= 0) & (Ct <= 1)))
        
class TestLS2Dv(unittest.TestCase):
    def test_field_shape(self):
        # Test that the single-element pressure field has the correct shape.
        b = 1.0
        f = 5
        c = 1480
        e = 0.0  # element centered at 0
        # Create a small mesh grid
        x = np.linspace(-10, 10, 50)
        z = np.linspace(1, 50, 50)
        xx, zz = np.meshgrid(x, z)
        field = run_ls_2Dv_service(b, f, c, e, xx, zz)
        self.assertEqual(field.shape, xx.shape)
        self.assertTrue(np.iscomplexobj(field))
        
class TestFullModeling(unittest.TestCase):
    def setUp(self):
        # Common parameters for full modeling tests.
        self.f = 5       # MHz
        self.c = 1480    # m/s
        self.M = 32
        self.dl = 0.5
        self.gd = 0.1
        self.F = np.inf
        self.wtype = 'rect'
        # Create a mesh grid for the field (in mm)
        self.z = np.linspace(1, 100 * self.dl, 100)
        self.x = np.linspace(-50 * self.dl, 50 * self.dl, 100)
        self.xx, self.zz = np.meshgrid(self.x, self.z)
    
    def test_full_modeling_output(self):
        # Run the full modeling service and check the output.
        p, A, d, g, e = run_mls_array_modeling_service(
            self.f, self.c, self.M, self.dl, self.gd, 20.0, self.F, self.wtype, self.xx, self.zz
        )
        # Check that the pressure field has the same shape as the mesh.
        self.assertEqual(p.shape, self.xx.shape)
        self.assertTrue(np.iscomplexobj(p))
        
    def test_beam_steering_effect(self):
        # Verify that changing the steering angle produces a different pressure field.
        p1, _, _, _, _ = run_mls_array_modeling_service(
            self.f, self.c, self.M, self.dl, self.gd, 20.0, self.F, self.wtype, self.xx, self.zz
        )
        p2, _, _, _, _ = run_mls_array_modeling_service(
            self.f, self.c, self.M, self.dl, self.gd, 30.0, self.F, self.wtype, self.xx, self.zz
        )
        # Compute the norm of the difference.
        diff_norm = np.linalg.norm(p1 - p2)
        # The fields should differ by a noticeable amount.
        self.assertGreater(diff_norm, 1e-3, "Pressure fields should differ with different steering angles")

if __name__ == '__main__':
    unittest.main()
