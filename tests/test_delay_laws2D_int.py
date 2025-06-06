# tests/test_delay_laws2D_int.py

import os
import subprocess
import numpy as np
import pytest
from application.delay_laws2D_int_service import (
    run_delay_laws2D_int_service,
    run_delay_laws2D_int_service_with_rays
)

def test_steering_only_case():
    """
    Test that in the steering-only case (DF = inf), the delay array:
      - Has shape (M,)
      - Contains only finite values.
      - Remains consistent with different plotting options.
    """
    M = 32
    s = 1.0
    angt = 0
    ang20 = 30
    DT0 = 25.4
    DF = float('inf')
    c1 = 1480
    c2 = 5900

    for plt_option in ['y', 'n']:
        delays = run_delay_laws2D_int_service(M, s, angt, ang20, DT0, DF, c1, c2, plt_option)
        assert delays.shape == (M,)
        assert np.all(np.isfinite(delays))

def test_focusing_case():
    """
    Test that in the focusing case (finite DF), the delay array:
      - Has shape (M,)
      - Contains non-negative values.
    """
    M = 32
    s = 1.0
    angt = 0
    ang20 = 30
    DT0 = 25.4
    DF = 8.0  # Focusing case.
    c1 = 1480
    c2 = 5900
    delays = run_delay_laws2D_int_service(M, s, angt, ang20, DT0, DF, c1, c2, 'n')
    assert delays.shape == (M,)
    assert np.all(delays >= 0)

def test_invalid_M():
    """
    Test that providing a non-positive M raises a ValueError.
    """
    M = 0
    s = 1.0
    angt = 0
    ang20 = 30
    DT0 = 25.4
    DF = 8.0
    c1 = 1480
    c2 = 5900
    with pytest.raises(ValueError, match="Number of elements M must be greater than zero."):
        run_delay_laws2D_int_service(M, s, angt, ang20, DT0, DF, c1, c2, 'n')

def test_invalid_interface_conditions():
    """
    Test that an invalid interface configuration (c1/c2 * sin(ang20) > 1) raises a ValueError.
    """
    M = 32
    s = 1.0
    angt = 0
    ang20 = 90  # Physically impossible case when c1 > c2
    DT0 = 25.4
    DF = 10
    c1 = 5900  # Making c1 > c2 intentionally incorrect
    c2 = 1480
    with pytest.raises(ValueError, match="Invalid input: \\(c1/c2\\) \\* sin\\(ang20\\) = .* > 1."):
        run_delay_laws2D_int_service(M, s, angt, ang20, DT0, DF, c1, c2, 'n')

def test_steering_angle_effect():
    """
    Verify that changing the steering angle (angt) results in different delay values.
    """
    M = 32
    s = 1.0
    angt1 = 0
    angt2 = 10
    ang20 = 30
    DT0 = 25.4
    DF = float('inf')
    c1 = 1480
    c2 = 5900
    delays1 = run_delay_laws2D_int_service(M, s, angt1, ang20, DT0, DF, c1, c2, 'n')
    delays2 = run_delay_laws2D_int_service(M, s, angt2, ang20, DT0, DF, c1, c2, 'n')
    assert not np.allclose(delays1, delays2)

def test_MATLAB_equivalent_case():
    """
    Verify the MATLAB-equivalent case with M=16, s=0.5, angt=5, ang20=60, DT0=10, DF=10.
    This case uses a stem plot for visualization.
    """
    M = 16
    s = 0.5
    angt = 5
    ang20 = 60
    DT0 = 10
    DF = 10
    c1 = 1480
    c2 = 5900
    delays = run_delay_laws2D_int_service(M, s, angt, ang20, DT0, DF, c1, c2, 'y')
    assert delays.shape == (M,)
    assert np.all(delays >= 0)

def test_plot_type_support():
    """
    Ensure that both 'line' and 'stem' plot types are correctly supported.
    """
    M = 16
    s = 0.5
    angt = 5
    ang20 = 60
    DT0 = 10
    DF = 10
    c1 = 1480
    c2 = 5900

    for plot_type in ["line", "stem"]:
        delays = run_delay_laws2D_int_service(M, s, angt, ang20, DT0, DF, c1, c2, 'y')
        assert delays.shape == (M,)
        assert np.all(delays >= 0)

def test_delays_and_rays_steering_only():
    """
    Test that in the steering-only case using the new service method,
    the function returns a tuple (delays, (xp, yp)) where:
      - delays has shape (M,)
      - xp and yp have shape (3, M) representing the three key points per element.
      - The delays match those from the basic service.
    """
    M = 32
    s = 1.0
    angt = 0
    ang20 = 30
    DT0 = 25.4
    DF = float('inf')
    c1 = 1480
    c2 = 5900
    delays_basic = run_delay_laws2D_int_service(M, s, angt, ang20, DT0, DF, c1, c2, 'n')
    delays, (xp, yp) = run_delay_laws2D_int_service_with_rays(M, s, angt, ang20, DT0, DF, c1, c2, 'n')
    assert delays.shape == (M,)
    np.testing.assert_allclose(delays, delays_basic, rtol=1e-6)
    assert xp.shape == (3, M)
    assert yp.shape == (3, M)

def test_delays_and_rays_focusing():
    """
    Test that in the focusing case using the new service method,
    the function returns a tuple (delays, (xp, yp)) where:
      - delays has shape (M,) and are non-negative.
      - xp and yp have shape (3, M) representing the three key points per element.
      - The focal point (third point) is consistent across all elements.
    """
    M = 16
    s = 0.5
    angt = 5
    ang20 = 60
    DT0 = 10
    DF = 10
    c1 = 1480
    c2 = 5900
    delays, (xp, yp) = run_delay_laws2D_int_service_with_rays(M, s, angt, ang20, DT0, DF, c1, c2, 'y')
    assert delays.shape == (M,)
    assert np.all(delays >= 0)
    assert xp.shape == (3, M)
    assert yp.shape == (3, M)
    # In the focusing case, the third point (focal point) should be identical for all elements.
    focal_x = xp[2, 0]
    focal_y = yp[2, 0]
    np.testing.assert_allclose(xp[2, :], focal_x, rtol=1e-6)
    np.testing.assert_allclose(yp[2, :], focal_y, rtol=1e-6)

if __name__ == "__main__":
    pytest.main()
