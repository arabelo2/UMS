# tests/test_cli_utils.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
import argparse
from interface.cli_utils import safe_eval, safe_float, parse_array

class TestCliUtils(unittest.TestCase):

    def test_safe_eval_valid_expressions(self):
        """Test safe_eval with valid arithmetic expressions."""
        self.assertAlmostEqual(safe_eval("3.14"), 3.14)
        self.assertAlmostEqual(safe_eval("6.35/2"), 3.175)
        self.assertAlmostEqual(safe_eval("10+5"), 15)
        self.assertAlmostEqual(safe_eval("10-3*2"), 4)
        self.assertAlmostEqual(safe_eval("2^3"), 8)  # ^ is now supported as exponentiation
        self.assertAlmostEqual(safe_eval("-1.5"), -1.5)

    def test_safe_eval_invalid_expressions(self):
        """Test safe_eval with invalid expressions."""
        with self.assertRaises(argparse.ArgumentTypeError):
            safe_eval("abc")  # Non-numeric
        with self.assertRaises(argparse.ArgumentTypeError):
            safe_eval("__import__('os').system('ls')")  # Disallowed built-in
        with self.assertRaises(argparse.ArgumentTypeError):
            safe_eval("x + 2")  # x is undefined

    def test_safe_float_valid(self):
        """Test safe_float with valid inputs."""
        self.assertAlmostEqual(safe_float("3.14"), 3.14)
        self.assertAlmostEqual(safe_float("6.35/2"), 3.175)

    def test_safe_float_invalid(self):
        """Test safe_float with invalid inputs."""
        with self.assertRaises(argparse.ArgumentTypeError):
            safe_float("abc")  # Not a valid float or arithmetic expression

    def test_parse_array_explicit_values(self):
        """Test parse_array with explicit values (more than 3 numbers -> explicit array)."""
        arr = parse_array("5,10,15,20")
        np.testing.assert_array_equal(arr, np.array([5,10,15,20], dtype=float))

    def test_parse_array_linspace(self):
        """Test parse_array with exactly three numbers interpreted as linspace parameters."""
        arr = parse_array("0,10,6")
        # Should be equivalent to np.linspace(0, 10, 6): [0, 2, 4, 6, 8, 10]
        expected = np.linspace(0, 10, 6)
        np.testing.assert_allclose(arr, expected, rtol=1e-6)

    def test_parse_array_negative_start_stop(self):
        """Test parse_array with negative start or stop values."""
        arr = parse_array("-10,10,5")
        expected = np.linspace(-10, 10, 5)
        np.testing.assert_allclose(arr, expected, rtol=1e-6)

    def test_parse_array_invalid_format(self):
        """Test parse_array with invalid format."""
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_array("abc,def,ghi")  # Non-numeric
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_array("10,,20")       # Empty entry
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_array("__import__('os')")

    def test_parse_array_arithmetic_expressions(self):
        """Test parse_array with arithmetic expressions inside.
        
        For input "6.35/2, 10+5, 3.14", since exactly three numbers are provided,
        they are interpreted as linspace parameters:
            start = 6.35/2 = 3.175, stop = 10+5 = 15, num_points = int(3.14) = 3.
        Expected result: np.linspace(3.175, 15, 3) -> [3.175, 9.0875, 15].
        """
        arr = parse_array("6.35/2, 10+5, 3.14")
        expected = np.linspace(3.175, 15, 3)
        np.testing.assert_allclose(arr, expected, rtol=1e-6)

    def test_parse_array_strip_quotes(self):
        """Ensure parse_array strips surrounding quotes.
        
        For input "'5,10,15'", since exactly three numbers are provided,
        it should be interpreted as linspace parameters: start=5, stop=10, num_points=15.
        Expected: np.linspace(5,10,15)
        """
        arr = parse_array("'5,10,15'")
        expected = np.linspace(5, 10, 15)
        np.testing.assert_allclose(arr, expected, rtol=1e-6)

    def test_parse_array_empty_string(self):
        """Test parse_array with an empty string, should raise error."""
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_array("")

if __name__ == '__main__':
    unittest.main()
