# tests/test_delay_laws2D_int.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from application.delay_laws2D_int_service import run_delay_laws2D_int_service

def test_steering_only_case():
    """
    Test that in the steering-only case (DF = inf), the delay array:
      - Has shape (M,)
      - Contains only finite values.
    """
    M = 32
    s = 1.0
    angt = 0
    ang20 = 30
    DT0 = 25.4
    DF = float('inf')
    c1 = 1480
    c2 = 5900
    delays = run_delay_laws2D_int_service(M, s, angt, ang20, DT0, DF, c1, c2, 'n')
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
    with pytest.raises(ValueError):
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

if __name__ == "__main__":
    pytest.main()
