# tests/test_T_fluid_solid.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
import numpy as np
from domain.T_fluid_solid import FluidSolidTransmission


# Fixture to close all plots after each test
@pytest.fixture(autouse=True)
def close_plots_after_test():
    """
    Fixture to close all matplotlib plots after each test.
    """
    yield
    plt.close('all')

@pytest.fixture
def test_params():
    """Provide standard test parameters."""
    return {
        "d1": 1.0, "cp1": 1500,  # Fluid properties
        "d2": 2.7, "cp2": 6000, "cs2": 3200,  # Solid properties
        "theta1": 30,  # Incident angle in degrees
    }

def test_valid_coefficients(test_params):
    """Test valid transmission coefficient calculations."""
    tpp, tps = FluidSolidTransmission.compute_coefficients(
        test_params["d1"], test_params["cp1"], test_params["d2"], test_params["cp2"], test_params["cs2"], test_params["theta1"]
    )
    assert np.isfinite(tpp) and np.isfinite(tps)

def test_critical_angle(test_params):
    """Test behavior at the critical angle."""
    theta_critical = np.degrees(np.arcsin(test_params["cp1"] / test_params["cp2"]))
    tpp, tps = FluidSolidTransmission.compute_coefficients(
        test_params["d1"], test_params["cp1"], test_params["d2"], test_params["cp2"], test_params["cs2"], theta_critical
    )
    assert np.isfinite(tpp) and np.isfinite(tps)

def test_total_internal_reflection(test_params):
    """Test behavior beyond the critical angle."""
    tpp, tps = FluidSolidTransmission.compute_coefficients(
        test_params["d1"], test_params["cp1"], test_params["d2"], test_params["cp2"], test_params["cs2"], 90
    )
    assert isinstance(tpp, (complex, np.complex128)) and isinstance(tps, (complex, np.complex128))

def test_zero_incident_angle(test_params):
    """Test when incident angle is 0 degrees."""
    tpp, tps = FluidSolidTransmission.compute_coefficients(
        test_params["d1"], test_params["cp1"], test_params["d2"], test_params["cp2"], test_params["cs2"], 0
    )
    assert np.isfinite(tpp) and np.isfinite(tps)

def test_invalid_input():
    """Test invalid inputs (negative wave speeds)."""
    with pytest.raises(ValueError):
        FluidSolidTransmission.compute_coefficients(1.0, -1500, 2.7, 6000, 3200, 30)
    with pytest.raises(ValueError):
        FluidSolidTransmission.compute_coefficients(1.0, 1500, 2.7, -6000, 3200, 30)
    with pytest.raises(ValueError):
        FluidSolidTransmission.compute_coefficients(1.0, 1500, 2.7, 6000, -3200, 30)

def test_vectorized_theta():
    """Test handling of vectorized incident angles."""
    theta_values = np.array([0, 15, 30, 45, 60, 75, 90])
    tpp, tps = FluidSolidTransmission.compute_coefficients(1.0, 1500, 2.7, 6000, 3200, theta_values)
    assert tpp.shape == theta_values.shape and tps.shape == theta_values.shape
    assert np.all(np.isfinite(tpp)) and np.all(np.isfinite(tps))

def test_near_zero_angle(test_params):
    """Test behavior for small angles approaching zero."""
    theta_near_zero = 1e-4
    tpp, tps = FluidSolidTransmission.compute_coefficients(
        test_params["d1"], test_params["cp1"], test_params["d2"], test_params["cp2"], test_params["cs2"], theta_near_zero
    )
    assert np.isfinite(tpp) and np.isfinite(tps)

def test_near_90_angle(test_params):
    """Test behavior for angles approaching 90 degrees."""
    theta_near_90 = 89.999
    tpp, tps = FluidSolidTransmission.compute_coefficients(
        test_params["d1"], test_params["cp1"], test_params["d2"], test_params["cp2"], test_params["cs2"], theta_near_90
    )
    assert np.isfinite(tpp) and np.isfinite(tps)
    assert isinstance(tpp, (complex, np.complex128)) and isinstance(tps, (complex, np.complex128))

if __name__ == "__main__":
    pytest.main()
