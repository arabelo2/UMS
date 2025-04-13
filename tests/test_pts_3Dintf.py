# tests/test_pts_3Dintf.py

import os
import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
from application.pts_3Dintf_service import run_pts_3Dintf_service

# Fixture to close all plots after each test
@pytest.fixture(autouse=True)
def close_plots_after_test():
    """
    Fixture to close all matplotlib plots after each test.
    """
    yield
    plt.close('all')

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
        
def test_pts_3Dintf_3d_input():
    """Test that run_pts_3Dintf_service correctly handles a full 3D grid."""
    # Create a 3D grid using meshgrid
    x = np.linspace(50, 60, 3)  # 3 points
    y = np.linspace(70, 80, 4)  # 4 points
    z = np.linspace(80, 90, 5)  # 5 points
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    xi = run_pts_3Dintf_service(0, 0, 1, 1, 30, 100, 1480, 5900, X, Y, Z)
    
    # The expected shape should match the grid shape: (3, 4, 5)
    expected_shape = (3, 4, 5)
    np.testing.assert_equal(xi.shape, expected_shape,
                            err_msg=f"Expected xi shape {expected_shape}, but got {xi.shape}")

if __name__ == "__main__":
    pytest.main()
