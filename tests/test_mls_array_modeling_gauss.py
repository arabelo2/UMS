# tests/test_mls_array_modeling_gauss.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from domain.mls_array_modeling_gauss import compute_pressure_field

def test_pressure_field_shape():
    """
    Test that the pressure field and coordinate arrays have the expected shapes.
    """
    # Test parameters (using defaults similar to the interface)
    f = 5.0
    c = 1480
    M = 32
    dl = 0.5
    gd = 0.1
    Phi = 20.0
    F = float('inf')
    window_type = 'rect'
    
    result = compute_pressure_field(f, c, M, dl, gd, Phi, F, window_type)
    p = result['p']
    x = result['x']
    z = result['z']
    
    # The spatial grid is 500 x 500
    assert p.shape == (500, 500)
    assert x.shape[0] == 500
    assert z.shape[0] == 500

def test_pressure_field_nonzero():
    """
    Test that the computed pressure field has non-zero values.
    """
    f = 5.0
    c = 1480
    M = 32
    dl = 0.5
    gd = 0.1
    Phi = 20.0
    F = float('inf')
    window_type = 'rect'
    
    result = compute_pressure_field(f, c, M, dl, gd, Phi, F, window_type)
    p = result['p']
    
    # At least some portion of the pressure field should be non-zero
    assert np.any(np.abs(p) > 0), "Pressure field should contain non-zero values."

if __name__ == "__main__":
    pytest.main()
