#!/usr/bin/env python3
# tests/test_pts_3Dintf.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
from application.pts_3Dintf_service import run_pts_3Dintf_service

@pytest.mark.parametrize(
    "ex, ey, xn, yn, angt, Dt0, c1, c2, x, y, z",
    [
        # Standard case
        (0, 0, 1, 1, 30, 100, 1480, 5900, 50, 50, 80),
        
        # Edge cases
        (0, 0, 0, 0, 30, 100, 1480, 5900, 50, 50, 80),  # No displacement
        (0, 0, 1, 1, 0, 100, 1480, 5900, 50, 50, 80),   # Angle 0 degrees
        (0, 0, 1, 1, 90, 100, 1480, 5900, 50, 50, 80),  # Angle 90 degrees
        (-5, -5, -1, -1, 45, 100, 1480, 5900, -50, -50, 80),  # Negative offsets
        (10, 10, 5, 5, 60, 200, 1500, 6000, 100, 100, 150),  # Large offsets
        (0, 0, 1, 1, 30, 100, 0.001, 10000, 50, 50, 80),  # Extreme wave speed ratio (c1 << c2)
        (0, 0, 1, 1, 30, 100, 10000, 0.001, 50, 50, 80),  # Extreme wave speed ratio (c1 >> c2)
        (0, 0, 1, 1, 30, 100, 1480, 5900, 1e-6, 1e-6, 1e-6),  # Very small values
        (0, 0, 1, 1, 30, 100, 1480, 5900, 1e6, 1e6, 1e6),  # Very large values
    ]
)
def test_pts_3Dintf_valid_cases(ex, ey, xn, yn, angt, Dt0, c1, c2, x, y, z):
    xi = run_pts_3Dintf_service(ex, ey, xn, yn, angt, Dt0, c1, c2, x, y, z)
    assert np.isfinite(xi), f"Expected finite value but got {xi}"

@pytest.mark.parametrize(
    "ex, ey, xn, yn, angt, Dt0, c1, c2, x, y, z",
    [
        (0, 0, 1, 1, 30, -100, 1480, 5900, 50, 50, 80),  # Negative Dt0
        (0, 0, 1, 1, 30, 100, -1480, 5900, 50, 50, 80),  # Negative c1 (invalid)
        (0, 0, 1, 1, 30, 100, 1480, -5900, 50, 50, 80),  # Negative c2 (invalid)
    ]
)
def test_pts_3Dintf_invalid_cases(ex, ey, xn, yn, angt, Dt0, c1, c2, x, y, z):
    with pytest.raises(ValueError):
        run_pts_3Dintf_service(ex, ey, xn, yn, angt, Dt0, c1, c2, x, y, z)

if __name__ == "__main__":
    pytest.main()
