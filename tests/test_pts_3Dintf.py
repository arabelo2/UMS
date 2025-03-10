# tests/test_pts_3Dintf.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
from application.pts_3Dintf_service import run_pts_3Dintf_service

@pytest.mark.parametrize(
    "ex, ey, xn, yn, angt, Dt0, c1, c2, x, y, z, expected_shape",
    [
        (0, 0, 1, 1, 30, 100, 1480, 5900, 50, 50, 80, ()),  # Scalar input
        (0, 0, 1, 1, 30, 100, 1480, 5900, [50, 60], [50, 60], [80, 90], (2,)),  # 1D array
        (0, 0, 1, 1, 30, 100, 1480, 5900, np.linspace(0, 100, 10), np.linspace(0, 100, 10), np.linspace(0, 100, 10), (10,)),  # Larger array
        (0, 0, 1, 1, 30, 100, 1480, 5900, np.array([[50, 60], [70, 80]]), np.array([[50, 60], [70, 80]]), np.array([[80, 90], [100, 110]]), (2, 2)),  # 2D array
    ]
)
def test_pts_3Dintf_valid_cases(ex, ey, xn, yn, angt, Dt0, c1, c2, x, y, z, expected_shape):
    """ Test that the function correctly computes `xi` with valid inputs. """
    xi = run_pts_3Dintf_service(ex, ey, xn, yn, angt, Dt0, c1, c2, x, y, z)

    # Ensure the output is finite
    assert np.isfinite(xi).all(), f"Expected finite values but got {xi}"

    # Check shape consistency
    if np.isscalar(xi):
        assert expected_shape == (), f"Expected scalar but got array of shape {xi.shape}"
    else:
        assert xi.shape == expected_shape, f"Expected shape {expected_shape}, got {xi.shape}"

@pytest.mark.parametrize(
    "ex, ey, xn, yn, angt, Dt0, c1, c2, x, y, z",
    [
        (0, 0, 1, 1, 30, -100, 1480, 5900, 50, 50, 80),  # Negative Dt0
        (0, 0, 1, 1, 30, 100, -1480, 5900, 50, 50, 80),  # Negative c1 (invalid)
        (0, 0, 1, 1, 30, 100, 1480, -5900, 50, 50, 80),  # Negative c2 (invalid)
    ]
)
def test_pts_3Dintf_invalid_cases(ex, ey, xn, yn, angt, Dt0, c1, c2, x, y, z):
    """ Ensure invalid input values raise ValueError. """
    with pytest.raises(ValueError):
        run_pts_3Dintf_service(ex, ey, xn, yn, angt, Dt0, c1, c2, x, y, z)

if __name__ == "__main__":
    pytest.main()
