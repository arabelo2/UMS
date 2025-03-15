# tests/test_delay_laws3Dint.py

import os
import subprocess
import numpy as np
import pytest

def run_cli(args):
    """
    Helper function to execute the CLI with given arguments.
    Returns stdout, stderr, and the exit code.
    """
    cmd = ["python", "interface/delay_laws3Dint_interface.py"] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout, result.stderr, result.returncode

@pytest.fixture(autouse=True)
def cleanup_plots():
    """
    Fixture to close any open matplotlib figures after each test.
    """
    import matplotlib.pyplot as plt
    yield
    plt.close('all')

def test_delay_steering_only():
    """
    Test delay laws in steering-only case (DF=inf).
    Expect the returned delay matrix to have shape (Mx, My) and contain finite values.
    """
    from domain.delay_laws3Dint import delay_laws3Dint
    td = delay_laws3Dint(Mx=4, My=4, sx=0.5, sy=0.5, thetat=20, phi=0,
                         theta2=45, DT0=10, DF=float('inf'), c1=1480, c2=5900, plt_option='n')
    assert td.shape == (4, 4)
    assert np.all(np.isfinite(td))

def test_delay_focusing():
    """
    Test delay laws in focusing case (DF finite).
    """
    from domain.delay_laws3Dint import delay_laws3Dint
    td = delay_laws3Dint(Mx=4, My=4, sx=0.5, sy=0.5, thetat=20, phi=0,
                         theta2=45, DT0=10, DF=10, c1=1480, c2=5900, plt_option='n')
    assert td.shape == (4, 4)
    # In focusing case, delays should be non-negative.
    assert np.all(td >= 0)

def test_cli_default_run():
    """
    Test CLI execution with default parameters (plot disabled).
    """
    stdout, stderr, code = run_cli(["--plt=n"])
    assert code == 0, f"CLI exited with code {code}. stderr: {stderr}"
    assert "Computed delays" in stdout

def test_cli_with_plot():
    """
    Test CLI execution with plotting enabled.
    """
    stdout, stderr, code = run_cli(["--plt=y"])
    assert code == 0, f"CLI exited with code {code}. stderr: {stderr}"
    assert "Computed delays" in stdout

if __name__ == "__main__":
    pytest.main()
