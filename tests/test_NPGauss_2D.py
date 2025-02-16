# tests/test_NPGauss_2D.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import subprocess
import os

def run_cli(args):
    """
    Helper function to call NPGauss_2D_interface.py with the provided args.
    Returns stdout, stderr, and exit code.
    """
    cmd = ["python", "interface/NPGauss_2D_interface.py"] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout, result.stderr, result.returncode

class TestNPGauss2DInterface(unittest.TestCase):

    def setUp(self):
        """
        Remove default output file if it exists before each test.
        """
        self.outfile = "np_gauss_2D_output.txt"
        if os.path.exists(self.outfile):
            os.remove(self.outfile)

    def test_default_run(self):
        """
        Runs with defaults:
         b=6, f=5, c=1500, e=0, x='-10,10,200', z=60, plot=Y
        Expect exit code=0, output file creation, and success message.
        """
        stdout, stderr, code = run_cli([])
        self.assertEqual(code, 0, f"CLI exited with code {code}, stderr: {stderr}")
        self.assertIn("Results saved to np_gauss_2D_output.txt", stdout)
        self.assertTrue(os.path.exists(self.outfile), "Output file not created with default run.")

    def test_plot_option_no(self):
        """
        Provide --plot N. Expect exit code=0, file creation, no crash.
        """
        stdout, stderr, code = run_cli(["--plot", "N"])
        self.assertEqual(code, 0, f"CLI exited with code {code}, stderr: {stderr}")
        self.assertIn("Results saved to np_gauss_2D_output.txt", stdout)
        self.assertTrue(os.path.exists(self.outfile))

    def test_custom_parameters(self):
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
        self.assertEqual(code, 0, f"CLI exited with code {code}, stderr: {stderr}")
        self.assertIn("Results saved to np_gauss_2D_output.txt", stdout)
        self.assertTrue(os.path.exists(self.outfile))

    def test_invalid_b_parameter(self):
        """
        b=abc => safe_float fails => exit code != 0, mention 'Invalid' in stderr.
        """
        stdout, stderr, code = run_cli(["--b", "abc"])
        self.assertNotEqual(code, 0, "CLI should fail with invalid b='abc'")
        self.assertIn("Invalid", stderr)

    def test_invalid_x_coordinates(self):
        """
        x=abc => parse_array fails => exit code != 0, mention 'Invalid format' in stderr.
        """
        stdout, stderr, code = run_cli(["--x", "abc"])
        self.assertNotEqual(code, 0, "CLI should fail with invalid x='abc'")
        self.assertIn("Invalid format", stderr)

    def test_zero_b_parameter(self):
        """
        b=0 => domain logic eventually leads to division by zero => exit code != 0,
        mention 'division by zero' or similar in stderr.
        """
        stdout, stderr, code = run_cli(["--b", "0", "--f", "5", "--c", "1500", "--e", "0", "--z", "60"])
        self.assertNotEqual(code, 0)
        self.assertIn("division by zero", stderr.lower())

    def test_zero_c_parameter(self):
        """
        c=0 => wave speed zero => domain logic or check => exit code != 0,
        mention 'division by zero' or similar in stderr.
        """
        stdout, stderr, code = run_cli(["--b", "6", "--f", "5", "--c", "0", "--z", "60"])
        self.assertNotEqual(code, 0)
        self.assertIn("division by zero", stderr.lower())

    def test_negative_offset(self):
        """
        Provide a negative e => valid usage, but let's confirm exit code=0, success message.
        """
        stdout, stderr, code = run_cli(["--b", "6", "--f", "5", "--c", "1500", "--e", "-5", "--x=-10,10,100", "--z", "60"])
        self.assertEqual(code, 0, f"CLI should handle negative offset e=-5, code={code}")
        self.assertIn("Results saved to np_gauss_2D_output.txt", stdout)
        self.assertTrue(os.path.exists(self.outfile))

if __name__ == '__main__':
    unittest.main()
