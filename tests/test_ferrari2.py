# tests/test_ferrari2.py

import os
import subprocess
import pytest
import numpy as np
import warnings
from domain.ferrari2 import ferrari2
from domain.interface2 import interface2

@pytest.fixture
def tolerance():
    return 1e-6  # Tolerance for numerical comparisons

@pytest.mark.parametrize("cr, DF, DT, DX, expected", [
    (1.0, 40.0, 30.0, 50.0, 50 * 30 / (40 + 30)),
    (1.0 + 1e-7, 40.0, 30.0, 50.0, 50 * 30 / (40 + 30)),
])
def test_identical_media(cr, DF, DT, DX, expected):
    """ Test when media are identical or nearly identical. """
    xi = ferrari2(cr, DF, DT, DX)
    assert np.isclose(xi, expected, atol=1e-6)

@pytest.mark.parametrize("cr, DF, DT, DX", [
    (1.5, 40.0, 30.0, 50.0),
    (2.0, 100.0, 10.0, 50.0),
    (1.2, 30.0, 20.0, 40.0),
    (1.8, 80.0, 15.0, 60.0),
])
def test_solution_satisfies_interface2(cr, DF, DT, DX, tolerance):
    """ Verify that the returned xi satisfies interface2. """
    xi = ferrari2(cr, DF, DT, DX)
    y_val = interface2(xi, cr, DF, DT, DX)
    assert np.isclose(y_val, 0.0, atol=tolerance)

@pytest.mark.parametrize("cr, DF, DT, DX", [
    (1.5, 40.0, 30.0, 50.0),
    (1.5, 40.0, 30.0, -50.0),
])
def test_positive_negative_DX(cr, DF, DT, DX):
    """ Test positive and negative DX values. """
    xi = ferrari2(cr, DF, DT, DX)
    assert DX >= 0 and 0 <= xi <= DX or DX < 0 and DX <= xi <= 0

@pytest.mark.parametrize("cr, DF, DT, DX", [
    (2.0, 100.0, 10.0, 50.0),
])
def test_fallback_branch_solution(cr, DF, DT, DX, tolerance):
    """ Test fallback branch with extreme values. """
    xi = ferrari2(cr, DF, DT, DX)
    y_val = interface2(xi, cr, DF, DT, DX)
    assert np.isclose(y_val, 0.0, atol=tolerance)
    assert 0 <= xi <= DX

@pytest.mark.parametrize("cr, DF, DT, DX", [
    (1.8, 1e6, 1e6, 1e6),
])
def test_large_values(cr, DF, DT, DX):
    """ Test stability with very large values. """
    xi = ferrari2(cr, DF, DT, DX)
    assert 0 <= xi <= DX

@pytest.mark.parametrize("cr, DF, DT, DX", [
    (1.5, np.array([40.0, 50.0, 60.0]), np.array([30.0, 40.0, 50.0]), np.array([50.0, 60.0, 70.0])),
])
def test_vectorized_inputs(cr, DF, DT, DX):
    """ Ensure ferrari2 can handle vectorized inputs. """
    xi = ferrari2(cr, DF, DT, DX)
    assert xi.shape == DF.shape
    assert np.all((xi >= 0) & (xi <= DX))

@pytest.mark.parametrize("cr, DF, DT, DX", [
    ("invalid", 40.0, 30.0, 50.0),
    (1.5, -40.0, 30.0, 50.0),
])
def test_invalid_input(cr, DF, DT, DX):
    """ Ensure invalid input values raise an error. """
    with pytest.raises(Exception):
        ferrari2(cr, DF, DT, DX)

def test_negative_DT():
    """
    Test that negative DT values are handled correctly by flipping DT and DX.
    """
    cr = 1.5
    DF = 40.0
    DT = -30.0
    DX = 50.0

    # Call ferrari2 and ensure it handles negative DT
    xi = ferrari2(cr, DF, DT, DX)

    # Verify that the result is valid
    assert np.isfinite(xi), "Result must be finite."
    assert 0 <= abs(xi) <= abs(DX), "Result must be within the expected range."

def test_complex_numbers():
    """
    Test that the function works correctly with complex numbers.
    """
    cr = 1.5
    DF = 40.0
    DT = -30.0 + 10.0j  # Complex DT
    DX = 50.0

    # Call ferrari2 and ensure it handles complex DT
    xi = ferrari2(cr, DF, DT, DX)

    # Verify that the result is valid and finite
    assert np.isfinite(xi), "Result must be finite."
    assert isinstance(xi, (float, complex)), "Result must be a number (float or complex)."

if __name__ == "__main__":
    warnings.simplefilter("ignore", RuntimeWarning)
    pytest.main()
