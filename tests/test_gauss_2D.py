# interface/test_gauss_2D.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import subprocess
import os

def run_cli(args):
    result = subprocess.run(
        ["python", "interface/gauss_2D_interface.py"] + args,
        capture_output=True, text=True
    )
    return result.stdout, result.stderr, result.returncode

class TestGauss2DInterface(unittest.TestCase):
    def setUp(self):
        self.outfile = "gauss_2D_output.txt"
        if os.path.exists(self.outfile):
            os.remove(self.outfile)

    def test_default_run(self):
        stdout, stderr, code = run_cli([])
        self.assertEqual(code, 0, f"CLI exited with code {code}, stderr: {stderr}")
        self.assertIn("Results saved to gauss_2D_output.txt", stdout)
        self.assertTrue(os.path.exists(self.outfile), "Output file was not created.")

    def test_plot_option_no(self):
        stdout, stderr, code = run_cli(["--plot", "N"])
        self.assertEqual(code, 0, f"CLI exited with code {code}, stderr: {stderr}")
        self.assertIn("Results saved to gauss_2D_output.txt", stdout)
        self.assertTrue(os.path.exists(self.outfile))

    def test_custom_parameters(self):
        """
        Provide a valid numeric b, f, c, z,
        and pass x1, x2 with no single quotes.
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
        self.assertIn("Results saved to gauss_2D_output.txt", stdout)
        self.assertTrue(os.path.exists(self.outfile))

    def test_invalid_b_parameter(self):
        stdout, stderr, code = run_cli(["--b", "abc"])
        self.assertNotEqual(code, 0)
        self.assertIn("Invalid", stderr)

    def test_invalid_x_coordinates(self):
        stdout, stderr, code = run_cli(["--x1", "abc"])
        self.assertNotEqual(code, 0)
        self.assertIn("Invalid format", stderr)

    def test_zero_b_parameter(self):
        stdout, stderr, code = run_cli(["--b", "0", "--f", "5", "--c", "1500", "--z", "60"])
        self.assertNotEqual(code, 0)
        self.assertIn("division by zero", stderr.lower())

    def test_zero_c_parameter(self):
        stdout, stderr, code = run_cli(["--b", "6", "--f", "5", "--c", "0", "--z", "60"])
        self.assertNotEqual(code, 0)
        self.assertIn("division by zero", stderr.lower())

if __name__ == '__main__':
    unittest.main()
