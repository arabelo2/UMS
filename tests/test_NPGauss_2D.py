# tests/test_NPGauss_2D.py

import os
import pytest
import subprocess
import matplotlib
matplotlib.use('Agg')  # Set the backend to non-interactive globally
import matplotlib.pyplot as plt

# Fixture to close all plots after each test
@pytest.fixture(autouse=True)
def close_plots_after_test():
    """
    Fixture to close all matplotlib plots after each test.
    """
    yield
    plt.close('all')

# Fixture to remove the output file before each test
@pytest.fixture(autouse=True)
def remove_output_file():
    """
    Fixture to remove the default output file before each test.
    """
    outfile = "np_gauss_2D_output.txt"
    if os.path.exists(outfile):
        os.remove(outfile)
    yield

def run_cli(args):
    """
    Helper function to execute the NPGauss_2D_interface.py CLI with given arguments.
    Returns stdout, stderr, and the exit code.
    """
    # Set the matplotlib backend in the subprocess environment
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"  # Use a non-interactive backend for matplotlib

    # Run the CLI script
    result = subprocess.run(
        ["python", "src/interface/NPGauss_2D_interface.py"] + args,
        capture_output=True,
        text=True,
        env=env,
    )
    return result.stdout, result.stderr, result.returncode

def test_default_run():
    """
    Runs with defaults:
     b=6, f=5, c=1500, e=0, x='-10,10,200', z=60, plot=Y
    Expect exit code=0, output file creation, and success message.
    """
    stdout, stderr, code = run_cli([])
    assert code == 0, f"CLI exited with code {code}, stderr: {stderr}"
    assert "Results saved to np_gauss_2D_output.txt" in stdout
    assert os.path.exists("np_gauss_2D_output.txt"), "Output file not created with default run."

def test_plot_option_no():
    """
    Provide --plot N. Expect exit code=0, file creation, no crash.
    """
    stdout, stderr, code = run_cli(["--plot", "N"])
    assert code == 0, f"CLI exited with code {code}, stderr: {stderr}"
    assert "Results saved to np_gauss_2D_output.txt" in stdout
    assert os.path.exists("np_gauss_2D_output.txt")

def test_custom_parameters():
    """
    Provide valid b, f, c, e, x, z, and test for exit code=0, file creation, etc.
    """
    args = [
        "--b", "6",
        "--f", "5",
        "--c", "1500",
        "--e", "2.5",
        "--x=-10,10,200",
        "--z", "80",
        "--plot", "N"
    ]
    stdout, stderr, code = run_cli(args)
    assert code == 0, f"CLI exited with code {code}, stderr: {stderr}"
    assert "Results saved to np_gauss_2D_output.txt" in stdout
    assert os.path.exists("np_gauss_2D_output.txt")

def test_invalid_b_parameter():
    """
    b=abc => safe_float fails => exit code != 0, mention 'Invalid' in stderr.
    """
    stdout, stderr, code = run_cli(["--b", "abc"])
    assert code != 0, "CLI should fail with invalid b='abc'"
    assert "Invalid" in stderr

def test_invalid_x_coordinates():
    """
    x=abc => parse_array fails => exit code != 0, mention 'Invalid format' in stderr.
    """
    stdout, stderr, code = run_cli(["--x", "abc"])
    assert code != 0, "CLI should fail with invalid x='abc'"
    assert "Invalid format" in stderr

def test_zero_b_parameter():
    """
    b=0 => domain logic eventually leads to division by zero => exit code != 0,
    mention 'division by zero' or similar in stderr.
    """
    stdout, stderr, code = run_cli(["--b", "0", "--f", "5", "--c", "1500", "--e", "0", "--z", "60"])
    assert code != 0
    assert "division by zero" in stderr.lower()

def test_zero_c_parameter():
    """
    c=0 => wave speed zero => domain logic or check => exit code != 0,
    mention 'division by zero' or similar in stderr.
    """
    stdout, stderr, code = run_cli(["--b", "6", "--f", "5", "--c", "0", "--z", "60"])
    assert code != 0
    assert "division by zero" in stderr.lower()

def test_negative_offset():
    """
    Provide a negative e => valid usage, but let's confirm exit code=0, success message.
    """
    stdout, stderr, code = run_cli(["--b", "6", "--f", "5", "--c", "1500", "--e", "-5", "--x=-10,10,100", "--z", "60"])
    assert code == 0, f"CLI should handle negative offset e=-5, code={code}"
    assert "Results saved to np_gauss_2D_output.txt" in stdout
    assert os.path.exists("np_gauss_2D_output.txt")

if __name__ == '__main__':
    pytest.main()
    