import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
from domain.on_axis_foc2D import OnAxisFocusedPiston

class TestOnAxisFocusedPiston(unittest.TestCase):
    def setUp(self):
        """Initialize solver with test parameters."""
        self.solver = OnAxisFocusedPiston(b=6, R=100, f=5, c=1480)

    def test_pressure_single_point(self):
        """Test pressure computation at a single depth."""
        z = np.array([50])
        p = self.solver.compute_pressure(z)
        self.assertTrue(np.isfinite(p).all(), "Pressure should be finite.")

    def test_pressure_multiple_points(self):
        """Test pressure computation over multiple depths."""
        z = np.linspace(20, 400, 10)
        p = self.solver.compute_pressure(z)
        self.assertEqual(p.shape, z.shape, "Output shape should match input shape.")

if __name__ == "__main__":
    unittest.main()
