# tests/test_interface2.py

import os
import subprocess
import pytest
import numpy as np
from domain.interface2 import interface2

@pytest.mark.parametrize("x, cr, df, dp, dpf, expected", [
    (10.0, 1.5, 20.0, 30.0, 40.0, 10.0 / np.sqrt(10.0**2 + 30.0**2) - 1.5 * (40.0 - 10.0) / np.sqrt((40.0 - 10.0)**2 + 20.0**2)),
])
def test_scalar_normal(x, cr, df, dp, dpf, expected):
    y = interface2(x, cr, df, dp, dpf)
    assert np.isclose(y, expected, atol=1e-6)

@pytest.mark.parametrize("x, cr, df, dp, dpf, expected", [
    (0.0, 1.2, 25.0, 35.0, 45.0, -1.2 * 45.0 / np.sqrt(45.0**2 + 25.0**2)),
])
def test_scalar_x_zero_dp_nonzero(x, cr, df, dp, dpf, expected):
    y = interface2(x, cr, df, dp, dpf)
    assert np.isclose(y, expected, atol=1e-6)

@pytest.mark.parametrize("x, cr, df, dp, dpf", [
    (0.0, 1.2, 25.0, 0.0, 45.0),
])
def test_scalar_x_zero_dp_zero(x, cr, df, dp, dpf):
    y = interface2(x, cr, df, dp, dpf)
    assert np.isnan(y), "Expected NaN when x=0 and dp=0 (division by zero)"

@pytest.mark.parametrize("x, cr, df, dp, dpf", [
    (np.array([-5, 0, 5, 10]), 1.3, 30.0, 40.0, 50.0),
])
def test_array_input_mixed_values(x, cr, df, dp, dpf):
    y = interface2(x, cr, df, dp, dpf)
    assert y.shape == x.shape
    for i, xi in enumerate(x):
        term1 = xi / np.sqrt(xi**2 + dp**2)
        term2 = cr * (dpf - xi) / np.sqrt((dpf - xi)**2 + df**2)
        expected = term1 - term2
        if np.isnan(expected):
            assert np.isnan(y[i])
        else:
            assert np.isclose(y[i], expected, atol=1e-6)

@pytest.mark.parametrize("x, cr, df, dp, dpf, expected", [
    (1e6, 1.1, 20.0, 30.0, 50.0, 2.1),
])
def test_large_values(x, cr, df, dp, dpf, expected):
    y = interface2(x, cr, df, dp, dpf)
    assert np.isclose(y, expected, atol=1e-6)

@pytest.mark.parametrize("x, cr, df, dp, dpf", [
    (np.array([]), 1.0, 20.0, 30.0, 40.0),
])
def test_empty_array(x, cr, df, dp, dpf):
    y = interface2(x, cr, df, dp, dpf)
    assert y.size == 0

@pytest.mark.parametrize("x, cr, df, dp, dpf", [
    ("not a number", 1.2, 20.0, 30.0, 40.0),
])
def test_invalid_type(x, cr, df, dp, dpf):
    with pytest.raises(Exception):
        interface2(x, cr, df, dp, dpf)

if __name__ == "__main__":
    pytest.main()
