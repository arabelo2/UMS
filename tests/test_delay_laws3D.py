# tests/test_delay_laws3D.py

import os
import subprocess
import pytest

import os
import subprocess

def run_cli(args):
    """
    Helper function to execute the delay_laws3D_interface.py CLI with given arguments.
    Returns stdout, stderr, and the exit code.
    """
    # Set the matplotlib backend in the subprocess environment
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"  # Use a non-interactive backend for matplotlib

    # Run the CLI script
    result = subprocess.run(
        ["python", "src/interface/delay_laws3D_interface.py"] + args,
        capture_output=True,
        text=True,
        env=env,
    )
    return result.stdout, result.stderr, result.returncode

@pytest.fixture(autouse=True)
def cleanup_output_file():
    """
    Fixture to remove the output file before and after each test.
    """
    outfile = "delay_laws3D_output.txt"
    if os.path.exists(outfile):
        os.remove(outfile)
    yield
    if os.path.exists(outfile):
        os.remove(outfile)

def test_default_run():
    """
    Test running with default parameters.
    Expect exit code 0 and output file created.
    """
    stdout, stderr, code = run_cli([])
    assert code == 0, f"CLI exited with code {code}. stderr: {stderr}"
    assert "Time delays saved to" in stdout
    assert os.path.exists("delay_laws3D_output.txt")

def test_focusing_mode():
    """
    Provide a focusing scenario by setting F to a finite value.
    """
    stdout, stderr, code = run_cli(["--F", "11", "--plot", "N"])
    assert code == 0, f"Focusing mode failed with code {code}."
    assert "Time delays saved to" in stdout
    assert os.path.exists("delay_laws3D_output.txt")

def test_invalid_M_zero():
    """
    Test that M=0 raises a ValueError.
    """
    stdout, stderr, code = run_cli(["--M", "0"])
    assert code != 0, "CLI should fail when M is 0."
    assert "m and n must be" in stderr.lower(), f"Unexpected stderr: {stderr}"

def test_invalid_wave_speed_zero():
    """
    Test that c=0 raises a ValueError for division by zero.
    """
    stdout, stderr, code = run_cli(["--c", "0"])
    assert code != 0, "CLI should fail when c is zero."
    assert "zero" in stderr.lower(), f"Unexpected stderr: {stderr}"

def test_custom_parameters():
    """
    Test the CLI with a set of custom parameters.
    """
    stdout, stderr, code = run_cli([
        "--M", "10", "--N", "12", "--sx", "0.2", "--sy", "0.25",
        "--theta", "30", "--phi", "15", "--F", "inf", "--c", "1500", "--plot", "N"
    ])
    assert code == 0, f"CLI failed with custom parameters. stderr: {stderr}"
    assert "Time delays saved to" in stdout
    assert os.path.exists("delay_laws3D_output.txt")

def test_camera_parameters():
    """
    Test the CLI with custom camera viewing angles.
    """
    stdout, stderr, code = run_cli([
        "--elev", "30", "--azim", "45", "--plot", "N"
    ])
    assert code == 0, f"CLI failed with custom camera parameters. stderr: {stderr}"
    assert os.path.exists("delay_laws3D_output.txt")

def test_invalid_elev():
    """
    Test that providing an invalid elev value results in an error.
    """
    stdout, stderr, code = run_cli(["--elev", "invalid"])
    assert code != 0, "CLI should fail with an invalid elev value."
    assert "invalid" in stderr.lower() or "error" in stderr.lower(), f"Unexpected stderr: {stderr}"

def test_invalid_azim():
    """
    Test that providing an invalid azim value results in an error.
    """
    stdout, stderr, code = run_cli(["--azim", "invalid"])
    assert code != 0, "CLI should fail with an invalid azim value."
    assert "invalid" in stderr.lower() or "error" in stderr.lower(), f"Unexpected stderr: {stderr}"
