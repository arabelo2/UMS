# tests/test_mls_array_model_int.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from application.ml s_array_model_int_service import run_mls_array_model_int_service

def test_output_shape():
    """
    Test that the output pressure field and grid dimensions are as expected.
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
    x = result['x']
    z = result['z']
    
    # Default grid dimensions: 200 x 200.
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
    
    assert np.any(np.abs(p) > 0)

if __name__ == "__main__":
    pytest.main()
