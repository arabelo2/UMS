# tests/test_mls_array_modeling_gauss.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from application.mls_array_modeling_gauss_service import run_mls_array_modeling_gauss

def test_pressure_field_shape():
    """
    Test that the pressure field and coordinate arrays have the expected shapes.
    """
    # Test parameters using updated naming convention
    f = 5.0          # Frequency in MHz
    c = 1480.0       # Wave speed in m/sec
    M = 32           # Number of elements
    dl = 0.5         # Normalized element length (d / wavelength)
    gd = 0.1         # Normalized gap between elements (g / d)
    Phi = 20.0       # Steering angle in degrees
    F = float('inf') # Focal length (inf for no focusing)
    wtype = 'rect'   # Window type

    result = run_mls_array_modeling_gauss(f, c, M, dl, gd, Phi, F, wtype)
    p = result['p']
    x = result['x']
    z = result['z']

    # The spatial grid is 500 x 500 as defined in the domain function
    assert p.shape == (500, 500)
    assert x.shape[0] == 500
    assert z.shape[0] == 500

def test_pressure_field_nonzero():
    """
    Test that the computed pressure field has non-zero values.
    """
    f = 5.0
    c = 1480.0
    M = 32
    dl = 0.5
    gd = 0.1
    Phi = 20.0
    F = float('inf')
    wtype = 'rect'
    
    result = run_mls_array_modeling_gauss(f, c, M, dl, gd, Phi, F, wtype)
    p = result['p']
    
    # Check that some part of the pressure field is non-zero.
    assert np.any(np.abs(p) > 0), "Pressure field should contain non-zero values."

if __name__ == "__main__":
    pytest.main()
