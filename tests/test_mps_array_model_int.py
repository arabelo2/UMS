# tests/test_mps_array_model_int.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import subprocess
import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive globally
import matplotlib.pyplot as plt

# ---------------------------
# Helper function for CLI tests
# ---------------------------
def run_cli(args):
    """
    Helper function to execute the mps_array_model_int_interface.py CLI with given arguments.
    Returns stdout, stderr, and the exit code.
    """
    # Set the matplotlib backend in the subprocess environment
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"  # Force the Agg backend in the subprocess

    cmd = ["python", "interface/mps_array_model_int_interface.py"] + args
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return result.stdout, result.stderr, result.returncode

# Fixture to close all plots after each test
@pytest.fixture(autouse=True)
def close_plots_after_test():
    """
    Fixture to close all matplotlib plots after each test.
    """
    yield
    plt.close('all')

# Fixture to remove the output file before and after each test
@pytest.fixture(autouse=True)
def cleanup_output_files():
    """Cleanup generated output files after each test."""
    files = ["mps_array_model_int_output.txt"]
    yield
    for f in files:
        if os.path.exists(f):
            os.remove(f)

# ---------------------------
# Domain Layer Tests
# ---------------------------
def test_domain_compute_field_keys():
    """
    Test that compute_field() from MPSArrayModelInt returns a dictionary with keys 'p', 'x', and 'z'
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
    Test that the computed velocity field is non-negative.
    """
    from domain.mps_array_model_int import MPSArrayModelInt
    xs = np.linspace(-5, 20, 50)
    zs = np.linspace(1, 20, 50)
    model = MPSArrayModelInt(
        lx=0.15, ly=0.15, gx=0.05, gy=0.05,
        f=5, d1=1.0, cp1=1480, d2=7.9, cp2=5900, cs2=3200, wave_type="p",
        L1=11, L2=11, angt=10.217, Dt0=50.8,
        theta20=20, phi=0, DF=10,  # focusing mode with finite DF
        ampx_type="rect", ampy_type="rect",
        xs=xs, zs=zs, y=0
    )
    result = model.compute_field()
    p = result['p']
    assert np.all(np.isfinite(p)), "All pressure values must be finite."
    assert np.all(p >= 0), "Pressure field should be non-negative."

def test_domain_invalid_parameters():
    """
    Test that invalid parameters (e.g., negative lx) raise a ValueError.
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
    Test the service layer in focusing mode (finite DF) returns a correctly shaped field.
    """
    from application.mps_array_model_int_service import run_mps_array_model_int_service
    xs = np.linspace(-5, 20, 80)
    zs = np.linspace(1, 20, 60)
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
    Test running the CLI with default parameters (plot disabled).
    Expect exit code 0 and the output file to be created.
    """
    stdout, stderr, code = run_cli(["--plot=n"])
    assert code == 0, f"CLI exited with code {code}. stderr: {stderr}"
    assert "Results saved to" in stdout
    assert os.path.exists("mps_array_model_int_output.txt")

def test_cli_with_plot():
    """
    Test CLI execution with plotting enabled and custom z_scale.
    """
    stdout, stderr, code = run_cli(["--plot=y", "--z_scale=50"])
    assert code == 0, f"CLI with plot failed with code {code}. stderr: {stderr}"
    assert "Results saved to" in stdout

def test_cli_invalid_parameters():
    """
    Test CLI with invalid parameters (e.g., L1=0) to ensure failure.
    """
    stdout, stderr, code = run_cli(["--L1=0", "--plot=n"])
    assert code != 0, "CLI should fail when L1 is 0."
    assert "error" in stderr.lower() or "exception" in stderr.lower()

def test_cli_custom_parameters():
    """
    Test CLI with custom parameters and a custom evaluation grid.
    """
    stdout, stderr, code = run_cli([
        "--lx=0.2", "--ly=0.2", "--gx=0.1", "--gy=0.1",
        "--f=5", "--d1=1.0", "--c1=1480", "--d2=7.9", "--c2=5900", "--cs2=3200",
        "--type=p", "--L1=10", "--L2=12", "--angt=5", "--Dt0=25.4",
        "--theta20=30", "--phi=15", "--DF=inf",
        "--ampx_type=rect", "--ampy_type=rect",
        '--xs="-10,10,40"', '--zs="2,12,50"', "--plot=n"
    ])
    assert code == 0, f"CLI custom parameters failed. stderr: {stderr}"
    assert "Results saved to" in stdout
    assert os.path.exists("mps_array_model_int_output.txt")

if __name__ == "__main__":
    pytest.main()
