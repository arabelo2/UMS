import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
from domain.discrete_windows import DiscreteWindows

class TestDiscreteWindows(unittest.TestCase):
    def test_valid_windows(self):
        """Test that all valid window types generate correct outputs."""
        M = 10
        valid_windows = ["cos", "Han", "Ham", "Blk", "tri", "rect"]
        for window in valid_windows:
            weights = DiscreteWindows.generate_weights(M, window)
            self.assertEqual(len(weights), M, f"Window {window} should return {M} elements.")

    def test_invalid_window_type(self):
        """Test that an invalid window type raises a ValueError."""
        with self.assertRaises(ValueError):
            DiscreteWindows.generate_weights(10, "invalid")

    def test_m_too_small(self):
        """Test that M <= 1 raises a ValueError."""
        with self.assertRaises(ValueError):
            DiscreteWindows.generate_weights(1, "rect")

if __name__ == "__main__":
    unittest.main()
