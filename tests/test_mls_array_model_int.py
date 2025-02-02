import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
from application.mls_array_model_int_service import MLSArrayModelIntService

class TestMLSArrayModelInt(unittest.TestCase):
    """Unit tests for the MLSArrayModelIntService."""

    def setUp(self):
        """Initialize test parameters."""
        self.f = 5  # Frequency (MHz)
        self.d1 = 1.0  # Density of first medium (g/cm³)
        self.c1 = 1480  # Wavespeed (m/s) in first medium
        self.d2 = 7.9  # Density of second medium (g/cm³)
        self.c2 = 5900  # Wavespeed (m/s) in second medium
        self.M = 32  # Number of elements
        self.d = 0.25  # Element length (mm)
        self.g = 0.05  # Gap length (mm)
        self.angt = 0  # Array tilt angle (degrees)
        self.ang20 = 30.0  # Steering angle in second medium (degrees)
        self.DF = 8  # Focal depth (mm) (∞ for no focusing)
        self.DT0 = 25.4  # Distance of array from interface (mm)
        self.window_type = "rect"  # Type of amplitude weighting function

        # Initialize service
        self.service = MLSArrayModelIntService(
            self.f, self.d1, self.c1, self.d2, self.c2,
            self.M, self.d, self.g, self.angt, self.ang20,
            self.DF, self.DT0, self.window_type
        )

    def test_pressure_field_dimensions(self):
        """Test if computed pressure field has expected dimensions."""
        x = np.linspace(-5, 15, 100)
        z = np.linspace(1, 20, 80)
        pressure = self.service.compute_pressure(x, z)
        self.assertEqual(pressure.shape, (80, 100), "Pressure field dimensions are incorrect.")

    def test_element_positions(self):
        """Test if element centroids are correctly computed."""
        expected_positions = np.linspace(-((self.M - 1) / 2) * (self.d + self.g),
                                         ((self.M - 1) / 2) * (self.d + self.g), self.M)
        np.testing.assert_array_almost_equal(self.service.solver.e, expected_positions, decimal=6,
                                             err_msg="Element positions do not match expected values.")

    def test_time_delays(self):
        """Ensure time delays are computed correctly."""
        td = self.service.solver.td
        self.assertEqual(len(td), self.M, "Time delay array size mismatch.")
        self.assertAlmostEqual(td[0], 0, places=6, msg="First delay should be zero after normalization.")

    def test_amplitude_weights(self):
        """Ensure amplitude weights are correctly applied."""
        Ct = self.service.solver.Ct
        self.assertEqual(len(Ct), self.M, "Amplitude weight array size mismatch.")
        self.assertTrue(np.all(Ct >= 0), "Amplitude weights should be non-negative.")

    def test_multiple_window_types(self):
        """Test if different window types are correctly applied."""
        for window in ["rect", "Han", "Ham", "Blk", "tri"]:
            with self.subTest(window=window):
                service = MLSArrayModelIntService(
                    self.f, self.d1, self.c1, self.d2, self.c2,
                    self.M, self.d, self.g, self.angt, self.ang20,
                    self.DF, self.DT0, window
                )
                Ct = service.solver.Ct
                self.assertEqual(len(Ct), self.M, f"Window '{window}' returned incorrect weight array size.")

    def test_varying_steering_angles(self):
        """Test pressure computation for different steering angles."""
        for angle in [0, 15, 30, 45]:
            with self.subTest(angle=angle):
                service = MLSArrayModelIntService(
                    self.f, self.d1, self.c1, self.d2, self.c2,
                    self.M, self.d, self.g, self.angt, angle,
                    self.DF, self.DT0, self.window_type
                )
                x = np.linspace(-5, 15, 100)
                z = np.linspace(1, 20, 80)
                pressure = service.compute_pressure(x, z)
                self.assertEqual(pressure.shape, (80, 100), f"Pressure field mismatch for angle {angle}")

if __name__ == "__main__":
    unittest.main()