# tests/test_rs_2Dv.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
import sys
from io import StringIO
import argparse

# Import functions from the interface module.
# (Ensure that your interface directory contains an __init__.py file.)
from interface.rs_2Dv_interface import (
    safe_float,
    safe_eval,
    parse_z,
    main as interface_main
)
from application.rs_2Dv_service import run_rs_2Dv_service

# --- Tests for arithmetic conversion functions --- #

class TestArithmeticConversion(unittest.TestCase):
    def test_safe_float_valid_number(self):
        # Simple numeric strings should work.
        self.assertEqual(safe_float("3.14"), 3.14)
    
    def test_safe_float_arithmetic(self):
        # Arithmetic expressions should be correctly evaluated.
        self.assertAlmostEqual(safe_float("6.35/2"), 6.35/2)
        self.assertAlmostEqual(safe_float("10-2"), 8)
        self.assertAlmostEqual(safe_float("-3"), -3)
        self.assertAlmostEqual(safe_float("2**3"), 8)
    
    def test_safe_float_invalid(self):
        # An invalid numeric expression should raise an error.
        with self.assertRaises(Exception):
            safe_float("abc")
    
    def test_safe_float_invalid_arithmetic(self):
        # Using an unsupported operator or invalid syntax should raise an error.
        with self.assertRaises(Exception):
            safe_float("6.35//2")  # floor division not supported
        with self.assertRaises(Exception):
            safe_float("import os")  # not allowed

# --- Tests for the z-coordinate parser --- #

class TestParseZFunction(unittest.TestCase):
    def test_parse_z_linspace(self):
        # When exactly three numbers are provided, interpret as linspace parameters.
        z = parse_z("5,200,500")
        expected = np.linspace(5, 200, 500)
        np.testing.assert_allclose(z, expected)
    
    def test_parse_z_explicit(self):
        # When not exactly three numbers, treat as explicit z-values.
        z = parse_z("5,10,15,20")
        expected = np.array([5, 10, 15, 20])
        np.testing.assert_allclose(z, expected)
    
    def test_parse_z_quotes_stripping(self):
        # Extra surrounding quotes should be removed.
        z = parse_z("'5,10,15,20'")
        expected = np.array([5, 10, 15, 20])
        np.testing.assert_allclose(z, expected)
    
    def test_parse_z_invalid(self):
        # An invalid value in the comma-separated list should raise an error.
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_z("5,10,abc,20")

# --- Tests for the overall interface --- #

class TestInterfaceMain(unittest.TestCase):
    def setUp(self):
        # Backup the original sys.argv and sys.stdout.
        self.original_argv = sys.argv.copy()
        self.original_stdout = sys.stdout

    def tearDown(self):
        # Restore original sys.argv and sys.stdout.
        sys.argv = self.original_argv
        sys.stdout = self.original_stdout

    def test_interface_main_valid_args(self):
        # Simulate a valid command-line call.
        # Patch plt.show() so that it does not block the test.
        import matplotlib.pyplot as plt
        original_show = plt.show
        plt.show = lambda: None

        sys.argv = [
            "rs_2Dv_interface.py",
            "--b", "6.35/2",
            "--f", "5",
            "--c", "1500",
            "--e", "0",
            "--x", "0",
            "--z", "1,50,500",
            "--N", "50"
        ]
        captured_output = StringIO()
        sys.stdout = captured_output

        try:
            interface_main()
            output = captured_output.getvalue()
            # Check that the output includes a line with "Computed normalized pressure"
            self.assertIn("Computed normalized pressure", output)
        finally:
            # Restore plt.show.
            plt.show = original_show

    def test_interface_main_invalid_b(self):
        # Test that providing an invalid expression for --b causes the parser to error out.
        sys.argv = [
            "rs_2Dv_interface.py",
            "--b", "invalid_expr",
            "--f", "5",
            "--c", "1500",
            "--e", "0",
            "--x", "0"
        ]
        # Expect SystemExit because parser.error() is called.
        with self.assertRaises(SystemExit):
            interface_main()

    def test_interface_main_invalid_z(self):
        # Test that providing an invalid --z argument (e.g., containing a non-numeric part) causes an error.
        sys.argv = [
            "rs_2Dv_interface.py",
            "--b", "6.35/2",
            "--f", "5",
            "--c", "1500",
            "--e", "0",
            "--x", "0",
            "--z", "1,50,abc",  # "abc" is invalid.
            "--N", "50"
        ]
        with self.assertRaises(SystemExit):
            interface_main()

# --- Tests for the service layer --- #

class TestRS2DvService(unittest.TestCase):
    def test_service_returns_complex_array(self):
        # Use explicit parameters to call the service.
        b = 6.35/2
        f = 5
        c = 1500
        e = 0
        x = 0
        z = np.linspace(5, 200, 500)
        p = run_rs_2Dv_service(b, f, c, e, x, z, N=50)
        self.assertTrue(isinstance(p, np.ndarray))
        # Check that the returned array has the same number of elements as z.
        self.assertEqual(p.size, z.size)

if __name__ == '__main__':
    unittest.main()
