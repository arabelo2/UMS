# tests/test_delay_laws3Dint.py

import subprocess
import numpy as np
import pytest

import os
import subprocess

def run_cli(args):
    """
    Helper function to execute the delay_laws3Dint_interface.py CLI with given arguments.
    Returns stdout, stderr, and the exit code.
    """
    # Set the matplotlib backend in the subprocess environment
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"  # Use a non-interactive backend for matplotlib

    # Run the CLI script
    result = subprocess.run(
        ["python", "src/interface/delay_laws3Dint_interface.py"] + args,
        capture_output=True,
        text=True,
        env=env,
    )
    return result.stdout, result.stderr, result.returncode

@pytest.fixture(autouse=True)
def cleanup_plots():
    """
    Fixture to close any open matplotlib figures after each test.
    """
    import matplotlib.pyplot as plt
    yield
    plt.close('all')

# -------------------------------------------------------------------
# Domain Layer Tests
# -------------------------------------------------------------------
def test_domain_layer_steering_only():
    """
    Test the domain layer (delay_laws3Dint) with DF=inf (steering-only).
    Expect a (Mx, My) shaped array of finite delays.
    """
    from domain.delay_laws3Dint import delay_laws3Dint
    Mx, My = 4, 4
    td = delay_laws3Dint(
        Mx=Mx, My=My,
        sx=0.5, sy=0.5,
        theta=20, phi=0, theta20=45,
        DT0=10, DF=float('inf'),
        c1=1480, c2=5900,
        plt_option='n'
    )
    assert td.shape == (Mx, My)
    assert np.all(np.isfinite(td)), "All delays should be finite for steering-only."

def test_domain_layer_focusing():
    """
    Test the domain layer with a finite DF for focusing.
    Delays should be non-negative and shape = (Mx, My).
    """
    from domain.delay_laws3Dint import delay_laws3Dint
    Mx, My = 4, 4
    td = delay_laws3Dint(
        Mx=Mx, My=My,
        sx=0.5, sy=0.5,
        theta=20, phi=0, theta20=45,
        DT0=10, DF=10,
        c1=1480, c2=5900,
        plt_option='n'
    )
    assert td.shape == (Mx, My)
    assert np.all(td >= 0), "Delays should be non-negative in focusing mode."

def test_domain_layer_invalid_c1c2_ratio():
    """
    If c1/c2 * sin(theta20) > 1, we expect the domain to clamp or raise an error.
    Here, we deliberately cause an out-of-range angle.
    """
    from domain.delay_laws3Dint import delay_laws3Dint
    Mx, My = 4, 4
    # c1=5900, c2=1480 => c1/c2= ~3.986; sin(theta20=45)=0.707 => ratio>1
    td = delay_laws3Dint(
        Mx=Mx, My=My,
        sx=0.5, sy=0.5,
        theta=20, phi=0, theta20=45,
        DT0=10, DF=10,
        c1=5900, c2=1480,
        plt_option='n'
    )
    # The code clamps arg=1 if ratio>1, so no error is raised. We can check if the result is finite.
    assert np.all(np.isfinite(td)), "The domain clamps the ratio, so we should still get finite delays."

def test_domain_layer_zero_elements():
    """
    If Mx=0 or My=0, we expect either an empty array or an error.
    Currently, code might produce an array of shape (0, My) or raise an error.
    We'll verify the behavior.
    """
    from domain.delay_laws3Dint import delay_laws3Dint
    with pytest.raises(ValueError):
        # We can decide that Mx=0 is invalid => raise ValueError in domain
        delay_laws3Dint(
            Mx=0, My=4,
            sx=0.5, sy=0.5,
            theta=20, phi=0, theta20=45,
            DT0=10, DF=10,
            c1=1480, c2=5900,
            plt_option='n'
        )

# -------------------------------------------------------------------
# Application Layer Tests
# -------------------------------------------------------------------
def test_application_layer_z_scale():
    """
    Test the service layer with a non-default z_scale. We expect the service to return
    both td and td_scaled. 
    """
    from application.delay_laws3Dint_service import run_delay_laws3Dint_service
    Mx, My = 4, 4
    td, td_scaled = run_delay_laws3Dint_service(
        Mx=Mx, My=My,
        sx=0.5, sy=0.5,
        theta=20, phi=0, theta20=45,
        DT0=10, DF=10,
        c1=1480, c2=5900,
        plt_option='n',
        view_elev=25, view_azim=20,
        z_scale=50.0
    )
    assert td.shape == (Mx, My)
    assert td_scaled.shape == (Mx, My)
    assert np.allclose(td_scaled, td * 50.0), "td_scaled should be td multiplied by z_scale=50."

def test_application_layer_return_values():
    """
    Ensure the service returns valid delays in both td and td_scaled.
    """
    from application.delay_laws3Dint_service import run_delay_laws3Dint_service
    Mx, My = 4, 4
    td, td_scaled = run_delay_laws3Dint_service(
        Mx=Mx, My=My,
        sx=0.5, sy=0.5,
        theta=0, phi=0, theta20=30,
        DT0=10, DF=float('inf'),
        c1=1480, c2=5900,
        plt_option='n',
        view_elev=25, view_azim=20,
        z_scale=2.0
    )
    assert td.shape == (Mx, My)
    assert td_scaled.shape == (Mx, My)
    assert np.all(td_scaled == td * 2.0), "Scaled delays should match raw delays * z_scale."

# -------------------------------------------------------------------
# Interface Layer Tests (CLI)
# -------------------------------------------------------------------
def test_cli_default_run_no_plot():
    """
    Test CLI execution with default parameters but no plot.
    """
    stdout, stderr, code = run_cli(["--plot=n"])
    assert code == 0, f"CLI exited with code {code}. stderr: {stderr}"
    assert "Computed delays:" in stdout
    assert "Results saved to mps_array_model_int_output.txt" in stdout

def test_cli_with_plot_and_z_scale():
    """
    Test CLI with a custom z_scale, verifying no error exit, 
    and presence of 'Computed delays' in stdout.
    """
    stdout, stderr, code = run_cli(["--z_scale=100", "--plot=y", "--elev=10", "--azim=60"])
    assert code == 0, f"CLI failed with code {code}. stderr: {stderr}"
    assert "Computed delays:" in stdout
    assert "Results saved to mps_array_model_int_output.txt" in stdout

def test_cli_invalid_input():
    """
    Test CLI with an invalid numeric input for L1 => we expect argparse to fail.
    """
    stdout, stderr, code = run_cli(["--L1=abc"])
    assert code != 0, "CLI should fail when L1 is non-numeric."
    assert "invalid int value" in stderr.lower()

def test_cli_focusing_mode():
    """
    Provide a focusing scenario by setting DF=5, ensuring no error, 
    and 'Computed delays' in stdout.
    """
    stdout, stderr, code = run_cli([
        "--DF=5", "--L1=4", "--L2=4", "--theta=10", "--phi=15", "--theta20=40",
        "--plot=n"
    ])
    assert code == 0, f"CLI focusing mode failed. stderr: {stderr}"
    assert "Computed delays:" in stdout
    assert "Results saved to mps_array_model_int_output.txt" in stdout

def test_cli_extreme_rotation():
    """
    Test extreme camera angles for the 3D stem plot, 
    verifying no error and presence of 'Computed delays'.
    """
    stdout, stderr, code = run_cli([
        "--plot=y", "--z_scale=50", 
        "--elev=85", "--azim=-45"
    ])
    assert code == 0, f"CLI extreme rotation failed. stderr: {stderr}"
    assert "Computed delays:" in stdout
    assert "Results saved to mps_array_model_int_output.txt" in stdout

def test_cli_big_z_scale():
    """
    Test an extremely large z_scale to verify no error and presence of 'Computed delays'.
    """
    stdout, stderr, code = run_cli([
        "--z_scale=1000", 
        "--plot=y"
    ])
    assert code == 0, f"CLI big z_scale test failed. stderr: {stderr}"
    assert "Computed delays:" in stdout
    assert "Results saved to mps_array_model_int_output.txt" in stdout

if __name__ == "__main__":
    pytest.main()
