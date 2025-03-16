# interface/test_gauss_2D.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import os
import unittest
import subprocess

def run_cli(args):
    """
    Helper function to execute the gauss_2D_interface.py CLI with given arguments.
    Returns stdout, stderr, and the exit code.
    """
    # Set the matplotlib backend in the subprocess environment
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"  # Use a non-interactive backend for matplotlib

    # Construct the absolute path to the script
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/interface/gauss_2D_interface.py'))

    # Run the CLI script
    result = subprocess.run(
        ["python", script_path] + args,
        capture_output=True,
        text=True,
        env=env,
    )
    return result.stdout, result.stderr, result.returncode

class TestGauss2DInterface(unittest.TestCase):

    def test_default_run(self):
        """
        Test the CLI with default parameters.
        """
        stdout, stderr, code = run_cli([])
        self.assertEqual(code, 0, f"CLI exited with code {code}, stderr: {stderr}")

    def test_plot_option_no(self):
        """
        Test the CLI with --plot N.
        """
        stdout, stderr, code = run_cli(["--plot", "N"])
        self.assertEqual(code, 0, f"CLI exited with code {code}, stderr: {stderr}")

    def test_custom_parameters(self):
        """
        Test the CLI with custom parameters.
        """
        args = [
            "--b", "6",
            "--f", "5",
            "--c", "1500",
            "--z", "60",
            "--x1=-10,10,200",
            "--x2=-10,10,40",
            "--plot", "Y"
        ]
        stdout, stderr, code = run_cli(args)
        self.assertEqual(code, 0, f"CLI exited with code {code}, stderr: {stderr}")

    def test_invalid_b_parameter(self):
        """
        Test the CLI with an invalid b parameter.
        """
        stdout, stderr, code = run_cli(["--b", "abc"])
        self.assertNotEqual(code, 0)
        self.assertIn("Invalid", stderr)

    def test_invalid_x_coordinates(self):
        """
        Test the CLI with invalid x coordinates.
        """
        stdout, stderr, code = run_cli(["--x1", "abc"])
        self.assertNotEqual(code, 0)
        self.assertIn("Invalid format", stderr)

    def test_zero_b_parameter(self):
        """
        Test the CLI with b=0.
        """
        stdout, stderr, code = run_cli(["--b", "0", "--f", "5", "--c", "1500", "--z", "60"])
        self.assertNotEqual(code, 0)
        self.assertIn("division by zero", stderr.lower())

    def test_zero_c_parameter(self):
        """
        Test the CLI with c=0.
        """
        stdout, stderr, code = run_cli(["--b", "6", "--f", "5", "--c", "0", "--z", "60"])
        self.assertNotEqual(code, 0)
        self.assertIn("division by zero", stderr.lower())

if __name__ == '__main__':
    unittest.main()
