# tests/test_mps_array_model_int.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import subprocess
import numpy as np
import pytest

# ---------------------------
# Helper function for CLI tests
# ---------------------------
def run_cli(args):
    """
    Helper function to execute the mps_array_model_int_interface.py CLI with given arguments.
    Returns stdout, stderr, and the exit code.
    """
    cmd = ["python", "interface/mps_array_model_int_interface.py"] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout, result.stderr, result.returncode

# ---------------------------
# Domain Layer Tests
# ---------------------------
def test_domain_compute_field_keys():
    """
    Test that compute_field() returns a dictionary with keys 'p', 'x', and 'z'
    and that the shapes are as expected.
    """
    from domain.mps_array_model_int import MPSArrayModelInt
    xs = np.linspace(-5, 20, 100)
    zs = np.linspace(1, 20, 100)
    model = MPSArrayModelInt(
        lx=0.15, ly=0.15, gx=0.05, gy=0.05,
        f=5, d1=1.0, cp1=1480, d2=7.9, cp2=5900, cs2=3200, wave_type="p",
        L1=11, L2=11, angt=10.217, Dt0=50.8,
        theta20=20, phi=0, DF=float('inf'),
        ampx_type="rect", ampy_type="rect",
        xs=xs, zs=zs, y=0
    )
    result = model.compute_field()
    assert isinstance(result, dict)
    for key in ['p', 'x', 'z']:
        assert key in result, f"Missing key: {key}"
    p = result['p']
    # p should have shape (len(zs), len(xs))
    assert p.shape == (len(zs), len(xs))
    np.testing.assert_allclose(result['x'], xs)
    np.testing.assert_allclose(result['z'], zs)

def test_domain_field_nonnegative():
    """
    In a valid configuration, the computed velocity magnitude should be non-negative.
    """
    from domain.mps_array_model_int import MPSArrayModelInt
    xs = np.linspace(-5, 20, 50)
    zs = np.linspace(1, 20, 50)
    model = MPSArrayModelInt(
        lx=0.15, ly=0.15, gx=0.05, gy=0.05,
        f=5, d1=1.0, cp1=1480, d2=7.9, cp2=5900, cs2=3200, wave_type="p",
        L1=11, L2=11, angt=10.217, Dt0=50.8,
        theta20=20, phi=0, DF=10,
        ampx_type="rect", ampy_type="rect",
        xs=xs, zs=zs, y=0
    )
    result = model.compute_field()
    p = result['p']
    assert np.all(p >= 0)

def test_domain_invalid_parameters():
    """
    Test that invalid parameters (e.g., negative element length) raise a ValueError.
    """
    from domain.mps_array_model_int import MPSArrayModelInt
    xs = np.linspace(-5, 20, 50)
    zs = np.linspace(1, 20, 50)
    with pytest.raises(ValueError):
        MPSArrayModelInt(
            lx=-0.15, ly=0.15, gx=0.05, gy=0.05,
            f=5, d1=1.0, cp1=1480, d2=7.9, cp2=5900, cs2=3200, wave_type="p",
            L1=11, L2=11, angt=10.217, Dt0=50.8,
            theta20=20, phi=0, DF=float('inf'),
            ampx_type="rect", ampy_type="rect",
            xs=xs, zs=zs, y=0
        )

# ---------------------------
# Application Layer Tests
# ---------------------------
def test_service_layer_returns_dict():
    """
    Test that the service layer returns a dictionary with keys 'p', 'x', and 'z'.
    """
    from application.mps_array_model_int_service import run_mps_array_model_int_service
    xs = np.linspace(-5, 20, 100)
    zs = np.linspace(1, 20, 100)
    result = run_mps_array_model_int_service(
        lx=0.15, ly=0.15, gx=0.05, gy=0.05,
        f=5, d1=1.0, c1=1480, d2=7.9, c2=5900, cs2=3200, wave_type="p",
        L1=11, L2=11, angt=10.217, Dt0=50.8,
        theta20=20, phi=0, DF=float('inf'),
        ampx_type="rect", ampy_type="rect",
        xs=xs, zs=zs, y=0
    )
    assert isinstance(result, dict)
    for key in ['p', 'x', 'z']:
        assert key in result

def test_service_layer_focusing():
    """
    Test the service layer in focusing mode (DF finite).
    """
    from application.mps_array_model_int_service import run_mps_array_model_int_service
    xs = np.linspace(-5, 20, 100)
    zs = np.linspace(1, 20, 100)
    result = run_mps_array_model_int_service(
        lx=0.15, ly=0.15, gx=0.05, gy=0.05,
        f=5, d1=1.0, c1=1480, d2=7.9, c2=5900, cs2=3200, wave_type="p",
        L1=11, L2=11, angt=10.217, Dt0=50.8,
        theta20=20, phi=0, DF=10,
        ampx_type="rect", ampy_type="rect",
        xs=xs, zs=zs, y=0
    )
    p = result['p']
    assert p.shape == (len(zs), len(xs))
    assert np.all(p >= 0)

# ---------------------------
# Interface Layer Tests (CLI)
# ---------------------------
def test_cli_default_run():
    """
    Test CLI execution with default parameters (plot disabled).
    """
    stdout, stderr, code = run_cli(["--plot=n"])
    assert code == 0, f"CLI exited with code {code}. stderr: {stderr}"
    assert "Results saved" in stdout

def test_cli_with_plot():
    """
    Test CLI execution with plotting enabled.
    """
    stdout, stderr, code = run_cli(["--plot=y", "--z_scale=50"])
    assert code == 0, f"CLI with plot failed with code {code}. stderr: {stderr}"
    assert "Results saved" in stdout

def test_cli_invalid_L1():
    """
    Test CLI with an invalid L1 (e.g., 0) and expect a non-zero exit code.
    """
    stdout, stderr, code = run_cli(["--L1=0", "--plot=n"])
    assert code != 0, "CLI should fail when L1 is 0."
    assert "error" in stderr.lower() or "exception" in stderr.lower()

if __name__ == "__main__":
    pytest.main()
