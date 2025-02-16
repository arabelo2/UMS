# interface/test_delay_laws2D.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import subprocess
import os

def run_cli(args):
    """
    Helper function that calls delay_laws2D_interface.py with the provided args.
    Returns stdout, stderr, and the exit code.
    """
    cmd = ["python", "interface/delay_laws2D_interface.py"] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout, result.stderr, result.returncode

class TestDelayLaws2DInterface(unittest.TestCase):

    def setUp(self):
        """
        Remove default output file if it exists before each test.
        """
        self.outfile = "delay_laws2D_output.txt"
        if os.path.exists(self.outfile):
            os.remove(self.outfile)

    def test_default_run(self):
        """
        Test running with default parameters:
         M=16, s=0.5, Phi=30, F=inf, c=1480, plot=Y
        Expect exit code=0 and output file created.
        """
        stdout, stderr, code = run_cli([])
        self.assertEqual(code, 0, f"CLI exited with code {code}, stderr={stderr}")
        self.assertIn("Time delays saved to delay_laws2D_output.txt", stdout)
        self.assertTrue(os.path.exists(self.outfile))

    def test_focusing_mode(self):
        """
        Provide a focusing scenario:
         M=16, s=0.5, Phi=0, F=15, c=1480, plot=Y
        Expect exit code=0, output file, and correct message.
        """
        stdout, stderr, code = run_cli(["--M", "16", "--s", "0.5", "--Phi", "0", 
                                        "--F", "15", "--c", "1480", "--plot", "Y"])
        self.assertEqual(code, 0, f"CLI exited with code {code}, stderr={stderr}")
        self.assertIn("Time delays saved to delay_laws2D_output.txt", stdout)
        self.assertTrue(os.path.exists(self.outfile))

    def test_plot_option_no(self):
        """
        Provide --plot N. We skip the plot but still generate the output file.
        """
        stdout, stderr, code = run_cli(["--plot", "N"])
        self.assertEqual(code, 0, f"CLI exited with code {code}, stderr={stderr}")
        self.assertIn("Time delays saved to delay_laws2D_output.txt", stdout)
        self.assertTrue(os.path.exists(self.outfile))

    def test_invalid_because_M_is_zero(self):
        """
        M=0 => domain logic raises ValueError: 'Number of elements M must be >= 1.'
        Expect non-zero exit code and error message in stderr.
        """
        stdout, stderr, code = run_cli(["--M", "0", "--s", "0.5", "--Phi", "0",
                                        "--F", "15", "--c", "1480"])
        self.assertNotEqual(code, 0)
        self.assertIn("must be >= 1", stderr.lower())

    def test_invalid_wave_speed_zero(self):
        """
        c=0 => domain logic raises ValueError: 'Wave speed c cannot be zero...'
        Expect non-zero exit code and mention of 'division by zero' or custom message in stderr.
        """
        stdout, stderr, code = run_cli(["--M", "16", "--s", "0.5", "--Phi", "30",
                                        "--F", "inf", "--c", "0"])
        self.assertNotEqual(code, 0)
        # The domain function typically raises 'Wave speed c cannot be zero (division by zero).'
        self.assertIn("division by zero", stderr.lower())

    def test_invalid_m_parameter_non_numeric(self):
        """
        Provide non-numeric M => we get an 'invalid int value' from argparse.
        Expect exit code != 0 and mention 'invalid int value' in stderr.
        """
        stdout, stderr, code = run_cli(["--M", "abc"])
        self.assertNotEqual(code, 0)
        self.assertIn("invalid int value", stderr.lower())

    def test_custom_parameters_exit_code(self):
        """
        A second scenario with custom parameters, verifying no error exit:
         M=10, s=1.0, Phi=-10, F=inf, c=1480 => Steering with negative angle
         Expect code=0, output file created, etc.
        """
        stdout, stderr, code = run_cli(["--M", "10", "--s", "1.0", "--Phi", "-10",
                                        "--F", "inf", "--c", "1480", "--plot", "N"])
        self.assertEqual(code, 0)
        self.assertIn("Time delays saved to delay_laws2D_output.txt", stdout)
        self.assertTrue(os.path.exists(self.outfile))

if __name__ == '__main__':
    unittest.main()
