# tests/test_T_fluid_solid.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
from domain.T_fluid_solid import FluidSolidTransmission

def test_valid_coefficients():
    """Test valid transmission coefficient calculations."""
    d1, cp1 = 1.0, 1500  # Fluid properties
    d2, cp2, cs2 = 2.7, 6000, 3200  # Solid properties
    theta1 = 30  # Incident angle in degrees
    
    tpp, tps = FluidSolidTransmission.compute_coefficients(d1, cp1, d2, cp2, cs2, theta1)
    assert np.isfinite(tpp) and np.isfinite(tps)

def test_critical_angle():
    """Test behavior at the critical angle."""
    d1, cp1 = 1.0, 1500
    d2, cp2, cs2 = 2.7, 6000, 3200
    theta1 = np.degrees(np.arcsin(cp1 / cp2))  # Critical angle
    
    tpp, tps = FluidSolidTransmission.compute_coefficients(d1, cp1, d2, cp2, cs2, theta1)
    assert np.isfinite(tpp) and np.isfinite(tps)

def test_total_internal_reflection():
    """Test behavior beyond the critical angle."""
    d1, cp1 = 1.0, 1500
    d2, cp2, cs2 = 2.7, 6000, 3200
    theta1 = 90  # Beyond critical angle

    tpp, tps = FluidSolidTransmission.compute_coefficients(d1, cp1, d2, cp2, cs2, theta1)
    assert isinstance(tpp, (complex, np.complex128)) and isinstance(tps, (complex, np.complex128))

def test_zero_incident_angle():
    """Test when incident angle is 0 degrees."""
    d1, cp1 = 1.0, 1500
    d2, cp2, cs2 = 2.7, 6000, 3200
    theta1 = 0  # Normal incidence
    
    tpp, tps = FluidSolidTransmission.compute_coefficients(d1, cp1, d2, cp2, cs2, theta1)
    assert np.isfinite(tpp) and np.isfinite(tps)

def test_invalid_input():
    """Test invalid inputs (negative wave speeds)."""
    with pytest.raises(ValueError):
        FluidSolidTransmission.compute_coefficients(1.0, -1500, 2.7, 6000, 3200, 30)
    with pytest.raises(ValueError):
        FluidSolidTransmission.compute_coefficients(1.0, 1500, 2.7, -6000, 3200, 30)
    with pytest.raises(ValueError):
        FluidSolidTransmission.compute_coefficients(1.0, 1500, 2.7, 6000, -3200, 30)

if __name__ == "__main__":
    pytest.main()
