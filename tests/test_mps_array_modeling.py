# tests/test_mps_array_modeling.py

import os
import subprocess
import numpy as np
import pytest
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive globally
import matplotlib.pyplot as plt

# ---------------------------
# Helper for CLI tests
# ---------------------------
def run_cli(args):
    """
    Helper function to execute the mps_array_modeling_interface.py CLI with given arguments.
    Returns stdout, stderr, and the exit code.
    """
    # Set the matplotlib backend in the subprocess environment
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"  # Use a non-interactive backend for matplotlib

    # Construct the absolute path to the script
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/interface/mps_array_modeling_interface.py'))

    # Run the CLI script
    result = subprocess.run(
        ["python", script_path] + args,
        capture_output=True,
        text=True,
        env=env,
    )
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
def cleanup_output_file():
    """
    Fixture to remove the output file before and after each test.
    """
    outfile = "mps_array_modeling_output.txt"
    if os.path.exists(outfile):
        os.remove(outfile)
    yield
    if os.path.exists(outfile):
        os.remove(outfile)

# ---------------------------
# Domain Layer Tests
# ---------------------------
def test_domain_layer():
    """
    Test the domain layer directly by instantiating the MPSArrayModeling class
    and verifying that the computed pressure field has the correct shape.
    """
    from domain.mps_array_modeling import MPSArrayModeling

    # Define a small evaluation grid
    xs = np.linspace(-10, 10, 50)
    zs = np.linspace(1, 15, 30)
    
    # Instantiate the model (using steering-only mode: F = inf)
    model = MPSArrayModeling(
        lx=0.15, ly=0.15, gx=0.05, gy=0.05,
        f=5, c=1480, L1=11, L2=11,
        theta=20, phi=0, F=float('inf'),
        ampx_type="rect", ampy_type="rect",
        xs=xs, zs=zs, y=0
    )
    p, xs_out, zs_out = model.compute_pressure_field()
    
    # p should be a complex array with shape (len(zs), len(xs))
    assert isinstance(p, np.ndarray)
    assert p.shape == (len(zs), len(xs))
    np.testing.assert_array_almost_equal(xs_out, xs)
    np.testing.assert_array_almost_equal(zs_out, zs)

# ---------------------------
# Application Layer Tests
# ---------------------------
def test_application_layer():
    """
    Test the application service layer to ensure integration with the domain.
    """
    from application.mps_array_modeling_service import run_mps_array_modeling_service

    xs = np.linspace(-10, 10, 50)
    zs = np.linspace(1, 15, 30)
    p, xs_out, zs_out = run_mps_array_modeling_service(
        lx=0.15, ly=0.15, gx=0.05, gy=0.05,
        f=5, c=1480, L1=11, L2=11,
        theta=20, phi=0, F=float('inf'),
        ampx_type="rect", ampy_type="rect",
        xs=xs, zs=zs, y=0
    )
    assert isinstance(p, np.ndarray)
    assert p.shape == (len(zs), len(xs))

# ---------------------------
# Interface Layer Tests (CLI)
# ---------------------------
def test_interface_default_run():
    """
    Test running the CLI with default parameters (plot disabled).
    Expect exit code 0 and the output file to be created.
    """
    stdout, stderr, code = run_cli(["--plot=n"])
    assert code == 0, f"CLI exited with code {code}. stderr: {stderr}"
    assert "Pressure field magnitude saved to" in stdout
    assert os.path.exists("mps_array_model_int_output.txt")

def test_interface_focusing_mode():
    """
    Test the CLI with a focusing scenario (DF finite).
    """
    stdout, stderr, code = run_cli(["--DF=15", "--plot=n"])
    assert code == 0, f"Focusing mode failed with exit code {code}. stderr: {stderr}"
    assert "Pressure field magnitude saved to" in stdout
    assert os.path.exists("mps_array_model_int_output.txt")

def test_interface_custom_parameters():
    """
    Test the CLI with a set of custom parameters and custom evaluation grid.
    """
    stdout, stderr, code = run_cli([
        "--lx=0.2", "--ly=0.2", "--gx=0.1", "--gy=0.1",
        "--f=5", "--c1=1480", "--c2=1480", "--L1=10", "--L2=12",
        "--theta=30", "--phi=15", "--DF=inf",
        "--ampx_type=rect", "--ampy_type=rect",
        '--xs="-10,10,40"', '--zs="2,12,50"', "--plot=n"
    ])
    assert code == 0, f"CLI failed with custom parameters. stderr: {stderr}"
    assert "Pressure field magnitude saved to" in stdout
    assert os.path.exists("mps_array_model_int_output.txt")

# ---------------------------
# Main Runner to Execute All Tests
# ---------------------------
if __name__ == "__main__":
    import pytest
    # Run all tests in the tests directory
    pytest.main(["tests"])
