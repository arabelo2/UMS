# tests/test_mls_array_model_int.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
from application.mls_array_model_int_service import run_mls_array_model_int_service

# Fixture to close all plots after each test
@pytest.fixture(autouse=True)
def close_plots_after_test():
    """
    Fixture to close all matplotlib plots after each test.
    """
    yield
    plt.close('all')

def test_output_shape_default_grid():
    """
    Test that the output pressure field and grid dimensions are as expected using default x and z.
    """
    f = 5
    d1 = 1.0
    c1 = 1480
    d2 = 7.9
    c2 = 5900
    M = 32
    d = 0.25
    g = 0.05
    angt = 0
    ang20 = 30
    DF = 8
    DT0 = 25.4
    wtype = 'rect'
    
    # Do not provide x and z; domain defaults will be used.
    result = run_mls_array_model_int_service(f, d1, c1, d2, c2, M, d, g, angt, ang20, DF, DT0, wtype)
    p = result['p']
    x = result['x']
    z = result['z']
    
    # Default grid is 200 x 200.
    assert p.shape == (200, 200)
    assert len(x) == 200
    assert len(z) == 200

def test_pressure_nonzero():
    """
    Test that the computed pressure field contains non-zero values.
    """
    f = 5
    d1 = 1.0
    c1 = 1480
    d2 = 7.9
    c2 = 5900
    M = 32
    d = 0.25
    g = 0.05
    angt = 0
    ang20 = 30
    DF = 8
    DT0 = 25.4
    wtype = 'rect'
    
    result = run_mls_array_model_int_service(f, d1, c1, d2, c2, M, d, g, angt, ang20, DF, DT0, wtype)
    p = result['p']
    
    # Ensure that at least some part of the pressure field is non-zero.
    assert np.any(np.abs(p) > 0)

def test_custom_grid_input():
    """
    Test that providing custom x and z arrays results in the output grid matching these arrays.
    """
    f = 5
    d1 = 1.0
    c1 = 1480
    d2 = 7.9
    c2 = 5900
    M = 32
    d = 0.25
    g = 0.05
    angt = 0
    ang20 = 30
    DF = 8
    DT0 = 25.4
    wtype = 'rect'
    
    # Define custom x and z.
    custom_x = np.linspace(-10, 10, 150)
    custom_z = np.linspace(5, 30, 150)
    
    result = run_mls_array_model_int_service(f, d1, c1, d2, c2, M, d, g, angt, ang20, DF, DT0, wtype, custom_x, custom_z)
    x = result['x']
    z = result['z']
    
    # Check that the returned x and z match the custom arrays.
    np.testing.assert_allclose(x, custom_x, rtol=1e-6)
    np.testing.assert_allclose(z, custom_z, rtol=1e-6)

if __name__ == "__main__":
    pytest.main()
