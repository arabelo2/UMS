#!/usr/bin/env python3
# tests/test_ps_3Dint.py

import os
import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
from domain.ps_3Dint import Ps3DInt

# Fixture to close all plots after each test
@pytest.fixture(autouse=True)
def close_plots_after_test():
    """
    Fixture to close all matplotlib plots after each test.
    """
    yield
    plt.close('all')

# Default parameters for testing
@pytest.fixture
def default_params():
    return {
        "lx": 6,
        "ly": 12,
        "f": 5,
        "mat": [1, 1480, 7.9, 5900, 3200, 'p'],
        "ex": 0,
        "ey": 0,
        "angt": 10.217,
        "Dt0": 50.8
    }

# Coordinates for testing
@pytest.fixture
def coordinates():
    return {
        "x": np.linspace(0, 30, 31).reshape(1, -1),  # Shape (1, 31)
        "z": np.linspace(0, 30, 31).reshape(1, -1),  # Shape (1, 31)
        "y": 0  # Scalar
    }

def test_default_parameters(default_params, coordinates):
    """Test with default parameters."""
    ps = Ps3DInt(**default_params)
    vx, vy, vz = ps.compute_velocity_components(coordinates["x"], coordinates["y"], coordinates["z"])

    # Check output shapes
    assert vx.shape == (1, 31)
    assert vy.shape == (1, 31)
    assert vz.shape == (1, 31)

    # Check output types
    assert isinstance(vx, np.ndarray)
    assert isinstance(vy, np.ndarray)
    assert isinstance(vz, np.ndarray)

    # Check velocity magnitude is non-negative
    v = np.sqrt(np.abs(vx)**2 + np.abs(vy)**2 + np.abs(vz)**2)
    assert np.all(v >= 0)

def test_angt_zero(default_params, coordinates):
    """Test with angt = 0 (array parallel to the interface)."""
    ps = Ps3DInt(**{**default_params, "angt": 0})
    vx, vy, vz = ps.compute_velocity_components(coordinates["x"], coordinates["y"], coordinates["z"])

    # Check output shapes
    assert vx.shape == (1, 31)
    assert vy.shape == (1, 31)
    assert vz.shape == (1, 31)

def test_angt_90(default_params, coordinates):
    """Test with angt = 90 (array perpendicular to the interface)."""
    ps = Ps3DInt(**{**default_params, "angt": 90})
    vx, vy, vz = ps.compute_velocity_components(coordinates["x"], coordinates["y"], coordinates["z"])

    # Check output shapes
    assert vx.shape == (1, 31)
    assert vy.shape == (1, 31)
    assert vz.shape == (1, 31)

def test_Dt0_zero_valid(default_params, coordinates):
    """Test with Dt0 = 0 (array on the interface) is now allowed and computes valid velocity components."""
    # We update the default so Dt0=0 is permitted.
    ps = Ps3DInt(**{**default_params, "Dt0": 0})
    vx, vy, vz = ps.compute_velocity_components(coordinates["x"], coordinates["y"], coordinates["z"])
    # Check output shapes and that the outputs are finite.
    assert vx.shape == coordinates["x"].shape
    assert np.all(np.isfinite(vx))
    assert np.all(np.isfinite(vy))
    assert np.all(np.isfinite(vz))

def test_f_zero(default_params, coordinates):
    """Test with f = 0 (zero frequency)."""
    with pytest.raises(ValueError):
        ps = Ps3DInt(**{**default_params, "f": 0})
        vx, vy, vz = ps.compute_velocity_components(coordinates["x"], coordinates["y"], coordinates["z"])

def test_lx_zero(default_params, coordinates):
    """Test with lx = 0 (zero element size in x-direction)."""
    with pytest.raises(ValueError):
        ps = Ps3DInt(**{**default_params, "lx": 0})
        vx, vy, vz = ps.compute_velocity_components(coordinates["x"], coordinates["y"], coordinates["z"])

def test_ly_zero(default_params, coordinates):
    """Test with ly = 0 (zero element size in y-direction)."""
    with pytest.raises(ValueError):
        ps = Ps3DInt(**{**default_params, "ly": 0})
        vx, vy, vz = ps.compute_velocity_components(coordinates["x"], coordinates["y"], coordinates["z"])

def test_wave_type_s(default_params, coordinates):
    """Test with wave_type = 's' (shear wave in the solid)."""
    ps = Ps3DInt(**{**default_params, "mat": [1, 1480, 7.9, 5900, 3200, 's']})
    vx, vy, vz = ps.compute_velocity_components(coordinates["x"], coordinates["y"], coordinates["z"])

    # Check output shapes
    assert vx.shape == (1, 31)
    assert vy.shape == (1, 31)
    assert vz.shape == (1, 31)

def test_c1_equal_c2(default_params, coordinates):
    """Test with c1 = c2 (identical wave speeds in both media)."""
    ps = Ps3DInt(**{**default_params, "mat": [1, 1480, 1, 1480, 3200, 'p']})
    vx, vy, vz = ps.compute_velocity_components(coordinates["x"], coordinates["y"], coordinates["z"])

    # Check output shapes
    assert vx.shape == (1, 31)
    assert vy.shape == (1, 31)
    assert vz.shape == (1, 31)

def test_invalid_mat(default_params, coordinates):
    """Test with invalid material properties."""
    with pytest.raises(ValueError):
        invalid_mat = [1, 1480, 7.9, 5900, 3200]  # Missing wave_type
        Ps3DInt(**{**default_params, "mat": invalid_mat})

def test_negative_parameters(default_params, coordinates):
    """Test with negative values for lx, ly, f, or Dt0."""
    with pytest.raises(ValueError):
        Ps3DInt(**{**default_params, "lx": -6})
    with pytest.raises(ValueError):
        Ps3DInt(**{**default_params, "ly": -12})
    with pytest.raises(ValueError):
        Ps3DInt(**{**default_params, "f": -5})
    with pytest.raises(ValueError):
        Ps3DInt(**{**default_params, "Dt0": -50.8})

if __name__ == '__main__':
    pytest.main()
