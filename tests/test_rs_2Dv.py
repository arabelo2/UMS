# tests/test_rs_2Dv.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
import sys
from io import StringIO
import argparse
import matplotlib.pyplot as plt

# Import helper functions from cli_utils.
from interface.cli_utils import safe_eval, safe_float, parse_array

# Import the main function from the CLI.
from interface.rs_2Dv_interface import main as cli_main

# Import the service layer for direct testing.
from application.rs_2Dv_service import run_rs_2Dv_service

class TestCliUtils(unittest.TestCase):
    def test_safe_eval_valid(self):
        self.assertAlmostEqual(safe_eval("6.35/2"), 6.35/2)
        self.assertEqual(safe_eval("10 - 2"), 8)
        self.assertEqual(safe_eval("-3"), -3)
        self.assertEqual(safe_eval("2**3"), 8)

    def test_safe_eval_invalid(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            safe_eval("invalid_expr")
        with self.assertRaises(argparse.ArgumentTypeError):
            safe_eval("6.35//2")
        with self.assertRaises(argparse.ArgumentTypeError):
            safe_eval("import os")

    def test_safe_float(self):
        self.assertEqual(safe_float("3.14"), 3.14)
        self.assertAlmostEqual(safe_float("6.35/2"), 6.35/2)
        with self.assertRaises(argparse.ArgumentTypeError):
            safe_float("abc")

    def test_parse_array_linspace(self):
        # Exactly three numbers: should use np.linspace.
        result = parse_array("5,200,500")
        expected = np.linspace(5, 200, 500)
        np.testing.assert_allclose(result, expected)
    
    def test_parse_array_explicit(self):
        # More (or fewer) than three numbers: treat as explicit values.
        result = parse_array("5,10,15,20")
        expected = np.array([5, 10, 15, 20])
        np.testing.assert_allclose(result, expected)
    
    def test_parse_array_invalid(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_array("5,abc,15,20")

class TestInterfaceMain(unittest.TestCase):
    def setUp(self):
        # Backup original sys.argv and sys.stdout.
        self.original_argv = sys.argv.copy()
        self.original_stdout = sys.stdout
        self.captured_output = StringIO()
        sys.stdout = self.captured_output
        # Patch plt.show() to a dummy function so that the tests do not block.
        self.original_show = plt.show
        plt.show = lambda: None

    def tearDown(self):
        sys.argv = self.original_argv
        sys.stdout = self.original_stdout
        plt.show = self.original_show

    def test_cli_main_valid_1D(self):
        # Test running the CLI in 1D mode.
        sys.argv = [
            "rs_2Dv_interface.py",
            "--b", "6.35/2",
            "--f", "5",
            "--c", "1500",
            "--e", "0",
            "--x", "0",
            "--z", "5,200,500",
            "--plot-mode", "1D"
        ]
        # We expect the CLI to run without raising SystemExit.
        try:
            cli_main()
        except SystemExit as e:
            self.fail("cli_main() raised SystemExit unexpectedly in 1D mode!")
        # Optionally, we could check that nothing was printed.
        output = self.captured_output.getvalue()
        # For our current CLI, no explicit message is printed on success.
        self.assertEqual(output, "")

    def test_cli_main_valid_2D(self):
        # Test running the CLI in 2D mode.
        sys.argv = [
            "rs_2Dv_interface.py",
            "--b", "6.35/2",
            "--f", "5",
            "--c", "1500",
            "--e", "0",
            "--x2=-10,10,200",  # Using equals sign to pass negative values properly.
            "--z2=1,20,200",
            "--plot-mode", "2D"
        ]
        try:
            cli_main()
        except SystemExit as e:
            self.fail("cli_main() raised SystemExit unexpectedly in 2D mode!")
        output = self.captured_output.getvalue()
        self.assertEqual(output, "")

    def test_cli_main_valid_both(self):
        # Test running the CLI in both mode.
        sys.argv = [
            "rs_2Dv_interface.py",
            "--b", "6.35/2",
            "--f", "5",
            "--c", "1500",
            "--e", "0",
            "--x", "0",
            "--z", "5,200,500",
            "--x2=-10,10,200",
            "--z2=1,20,200",
            "--plot-mode", "both"
        ]
        try:
            cli_main()
        except SystemExit as e:
            self.fail("cli_main() raised SystemExit unexpectedly in both mode!")
        output = self.captured_output.getvalue()
        self.assertEqual(output, "")

    def test_cli_main_invalid_b(self):
        # Test that providing an invalid arithmetic expression for --b causes a SystemExit.
        sys.argv = [
            "rs_2Dv_interface.py",
            "--b", "invalid_expr",
            "--f", "5",
            "--c", "1500",
            "--e", "0",
            "--x", "0"
        ]
        with self.assertRaises(SystemExit):
            cli_main()

    def test_cli_main_invalid_x2(self):
        # Test that providing an invalid array for --x2 causes a SystemExit.
        sys.argv = [
            "rs_2Dv_interface.py",
            "--b", "6.35/2",
            "--f", "5",
            "--c", "1500",
            "--e", "0",
            "--x2=abc,10,200",  # Invalid because "abc" is not a number.
            "--z2=1,20,200",
            "--plot-mode", "2D"
        ]
        with self.assertRaises(SystemExit):
            cli_main()

    def test_cli_main_invalid_z2(self):
        # Test that providing an invalid array for --z2 causes a SystemExit.
        sys.argv = [
            "rs_2Dv_interface.py",
            "--b", "6.35/2",
            "--f", "5",
            "--c", "1500",
            "--e", "0",
            "--x2=-10,10,200",
            "--z2=1,xyz,200",  # "xyz" is invalid.
            "--plot-mode", "2D"
        ]
        with self.assertRaises(SystemExit):
            cli_main()

class TestRS2DvService(unittest.TestCase):
    def test_service_returns_valid_output_1D(self):
        # Test that the service returns a NumPy array of the correct size for 1D simulation.
        b = 6.35/2
        f = 5
        c = 1500
        e = 0
        x = 0
        z = np.linspace(5, 200, 500)
        p = run_rs_2Dv_service(b, f, c, e, x, z, N=50)
        self.assertTrue(isinstance(p, np.ndarray))
        # After squeezing, the array should have size equal to the number of z points.
        self.assertEqual(np.prod(p.shape), 500)

if __name__ == '__main__':
    unittest.main()
