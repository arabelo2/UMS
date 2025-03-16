# tests/test_ps_3Dv.py

import os
import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive
import matplotlib.pyplot as plt
from application.ps_3Dv_service import run_ps_3Dv_service

# Fixture to close all plots after each test
@pytest.fixture(autouse=True)
def close_plots_after_test():
    """
    Fixture to close all matplotlib plots after each test.
    """
    yield
    plt.close('all')

# Helper functions to mimic the interface meshgrid creation for 2D and 3D cases.
def meshgrid_2d(x, y):
    # For a 2D simulation with (vector, vector, scalar), the interface uses:
    # np.meshgrid(np.atleast_1d(x), np.atleast_1d(y))
    X, Y = np.meshgrid(np.atleast_1d(x), np.atleast_1d(y))
    return X, Y

def meshgrid_3d(x, y, z):
    # For a full 3D simulation, use indexing='ij'
    X, Y, Z = np.meshgrid(np.atleast_1d(x), np.atleast_1d(y), np.atleast_1d(z), indexing='ij')
    return X, Y, Z

# Test parameters for all tests
lx = 6
ly = 12
f = 5
c = 1480
ex = 0
ey = 0

# --- 1D Simulation Tests ---
def test_1d_simulation():
    # x and y are scalars; z is a vector.
    x = 0
    y = 0
    z = np.linspace(5, 100, 50)
    p = run_ps_3Dv_service(lx, ly, f, c, ex, ey, x, y, z)
    # Expect p shape equal to z.shape.
    assert p.shape == z.shape
    assert np.iscomplexobj(p)

# --- 2D Simulation Tests ---
def test_2d_simulation_vector_vector_scalar():
    # (vector, vector, scalar): x and y are vectors, z is scalar.
    x = np.linspace(-5, 5, 20)
    y = np.linspace(-5, 5, 30)
    z = 50
    # Create meshgrid manually as the service expects pre-broadcastable arrays.
    X, Y = meshgrid_2d(x, y)  # Note: shape (30,20)
    p = run_ps_3Dv_service(lx, ly, f, c, ex, ey, X, Y, z)
    # Expected shape is (30,20)
    assert p.shape == X.shape
    assert np.iscomplexobj(p)

def test_2d_simulation_vector_scalar_vector():
    # (vector, scalar, vector): x is vector, y is scalar, z is vector.
    x = np.linspace(-5, 5, 25)
    y = 0
    z = np.linspace(5, 100, 40)
    # Create meshgrid for x and z.
    X, Z = meshgrid_2d(x, z)
    # Broadcast y (scalar) to shape of meshgrid.
    Y = np.full(X.shape, y)
    p = run_ps_3Dv_service(lx, ly, f, c, ex, ey, X, Y, Z)
    # Expected shape is (40,25)
    assert p.shape == X.shape
    assert np.iscomplexobj(p)

def test_2d_simulation_scalar_vector_vector():
    # (scalar, vector, vector): x is scalar, y and z are vectors.
    x = 0
    y = np.linspace(-5, 5, 30)
    z = np.linspace(5, 100, 35)
    Y, Z = meshgrid_2d(y, z)
    X = np.full(Y.shape, x)
    p = run_ps_3Dv_service(lx, ly, f, c, ex, ey, X, Y, Z)
    assert p.shape == Y.shape
    assert np.iscomplexobj(p)

# --- 3D Simulation Test ---
def test_3d_simulation():
    # (vector, vector, vector): all three are vectors.
    x = np.linspace(-5, 5, 10)
    y = np.linspace(-5, 5, 15)
    z = np.linspace(5, 100, 20)
    X, Y, Z = meshgrid_3d(x, y, z)
    p = run_ps_3Dv_service(lx, ly, f, c, ex, ey, X, Y, Z)
    # Expected shape is (len(x), len(y), len(z))
    assert p.shape == (len(x), len(y), len(z))
    assert np.iscomplexobj(p)

# --- Optional Integration Parameters Test ---
def test_optional_integration_params():
    # Compare default integration parameters with specified ones.
    x = 0
    y = 0
    z = np.linspace(5, 100, 50)
    p_default = run_ps_3Dv_service(lx, ly, f, c, ex, ey, x, y, z)
    p_manual = run_ps_3Dv_service(lx, ly, f, c, ex, ey, x, y, z, P=50, Q=50)
    # They need not be identical, but should be reasonably close.
    np.testing.assert_allclose(p_default, p_manual, rtol=0.2)

# --- Edge Cases ---
def test_inconsistent_coordinate_shapes():
    # Passing arrays that cannot be broadcast should raise an error.
    x = np.linspace(-5, 5, 10)  # shape (10,)
    y = np.linspace(-5, 5, 15)  # shape (15,)
    z = np.linspace(5, 100, 20) # shape (20,)
    # Without meshgrid, these shapes are not directly broadcastable.
    with pytest.raises(ValueError):
        # This should fail because np.broadcast cannot reconcile (10,), (15,), (20,)
        _ = run_ps_3Dv_service(lx, ly, f, c, ex, ey, x, y, z)

# --- 2D Simulation with 3D Field Plotting Mode ---
# While the --plot-3dfield flag affects plotting in the interface, the service function
# always returns p. Therefore, we simulate the expected shapes for the 3D field plot.
def test_2d_simulation_3dfield_vector_scalar_vector():
    # (vector, scalar, vector) simulation
    x = np.linspace(-5, 5, 25)
    y = 0
    z = np.linspace(5, 100, 40)
    X, Z = meshgrid_2d(x, z)
    Y = np.full(X.shape, y)
    p = run_ps_3Dv_service(lx, ly, f, c, ex, ey, X, Y, Z)
    # For plot-3dfield mode, the service output remains the same.
    # The interface would use a 3D scatter plot of (x, z, |p|) in this case.
    assert p.shape == X.shape
    assert np.iscomplexobj(p)

def test_2d_simulation_3dfield_scalar_vector_vector():
    # (scalar, vector, vector) simulation
    x = 0
    y = np.linspace(-5, 5, 30)
    z = np.linspace(5, 100, 35)
    Y, Z = meshgrid_2d(y, z)
    X = np.full(Y.shape, x)
    p = run_ps_3Dv_service(lx, ly, f, c, ex, ey, X, Y, Z)
    # For plot-3dfield mode, the interface would use a 3D scatter plot of (y, z, |p|).
    assert p.shape == Y.shape
    assert np.iscomplexobj(p)

if __name__ == '__main__':
    pytest.main()
