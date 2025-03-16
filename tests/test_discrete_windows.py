# tests/test_discrete_windows.py

import sys
import os
import unittest
import subprocess

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

def run_cli(args):
    """
    Helper function to execute the discrete_windows_interface.py CLI with given arguments.
    Returns stdout, stderr, and the exit code.
    """
    # Set the matplotlib backend in the subprocess environment
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"  # Use a non-interactive backend for matplotlib

    # Construct the absolute path to the script
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/interface/discrete_windows_interface.py'))

    # Run the CLI script
    result = subprocess.run(
        ["python", script_path] + args,
        capture_output=True,
        text=True,
        env=env,
    )
    return result.stdout, result.stderr, result.returncode

class TestDiscreteWindowsInterface(unittest.TestCase):

    def setUp(self):
        """
        Remove the default output file if it exists before each test.
        """
        self.outfile = "discrete_windows_output.txt"
        if os.path.exists(self.outfile):
            os.remove(self.outfile)

    def test_default_run(self):
        """
        Runs with defaults:
         M=16, type='Blk', plot='Y'
        Expect exit code=0, output file created, and success message.
        """
        stdout, stderr, code = run_cli([])
        self.assertEqual(code, 0, f"CLI exited with code {code}, stderr: {stderr}")
        self.assertIn("Window amplitudes saved to discrete_windows_output.txt", stdout)
        self.assertTrue(os.path.exists(self.outfile), "Output file not created on default run.")

    def test_plot_disabled(self):
        """
        Provide --plot N. We skip the plot but still output the file.
        """
        stdout, stderr, code = run_cli(["--plot", "N"])
        self.assertEqual(code, 0, f"CLI exited with code {code}, stderr: {stderr}")
        self.assertIn("Window amplitudes saved to discrete_windows_output.txt", stdout)
        self.assertTrue(os.path.exists(self.outfile))

    def test_custom_parameters(self):
        """
        Provide custom M=10, type='Han', and check for success.
        """
        stdout, stderr, code = run_cli(["--M", "10", "--type", "Han", "--plot", "N"])
        self.assertEqual(code, 0, f"CLI exited with code {code}, stderr={stderr}")
        self.assertIn("Window amplitudes saved to discrete_windows_output.txt", stdout)
        self.assertTrue(os.path.exists(self.outfile))

    def test_invalid_m_parameter_zero(self):
        """
        M=0 => domain function raises ValueError => exit code !=0, mention in stderr.
        """
        stdout, stderr, code = run_cli(["--M", "0"])
        self.assertNotEqual(code, 0, "M=0 should fail but got exit code 0.")
        self.assertIn("must be >= 1", stderr.lower())

    def test_negative_m_parameter(self):
        """
        M=-5 => domain function raises ValueError => exit code != 0, mention in stderr.
        """
        stdout, stderr, code = run_cli(["--M", "-5"])
        self.assertNotEqual(code, 0)
        self.assertIn("must be >= 1", stderr.lower())

    def test_invalid_m_parameter_non_numeric(self):
        """
        M=abc => argparse triggers error => exit code != 0, mention 'invalid int value' in stderr.
        """
        stdout, stderr, code = run_cli(["--M", "abc"])
        self.assertNotEqual(code, 0)
        self.assertIn("invalid int value", stderr.lower())

    def test_invalid_type_parameter(self):
        """
        Provide an invalid wtype => domain logic raises ValueError =>
        exit code != 0, mention 'Invalid window type' in stderr.
        """
        stdout, stderr, code = run_cli(["--type", "foo"])
        self.assertNotEqual(code, 0)
        self.assertIn("invalid window type", stderr.lower())

    def test_plot_option_yes(self):
        """
        Provide --plot Y. We only check that it doesn't crash and that the file is created.
        """
        stdout, stderr, code = run_cli(["--plot", "Y"])
        self.assertEqual(code, 0, f"CLI exited with code {code}, stderr={stderr}")
        self.assertIn("Window amplitudes saved to discrete_windows_output.txt", stdout)
        self.assertTrue(os.path.exists(self.outfile))

if __name__ == '__main__':
    unittest.main()
