# tests/test_ls_2Dv.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
import sys
from io import StringIO
import argparse
import matplotlib.pyplot as plt

from interface.cli_utils import safe_eval, safe_float, parse_array
from interface.ls_2Dv_interface import main as cli_main
from application.ls_2Dv_service import run_ls_2Dv_service

class TestLs2DvCliUtils(unittest.TestCase):
    def test_safe_eval_valid(self):
        self.assertAlmostEqual(safe_eval("6.35/2"), 6.35/2)
        self.assertEqual(safe_eval("10-2"), 8)
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
        result = parse_array("5,200,500")
        expected = np.linspace(5, 200, 500)
        np.testing.assert_allclose(result, expected)
    
    def test_parse_array_explicit(self):
        result = parse_array("5,10,15,20")
        expected = np.array([5, 10, 15, 20])
        np.testing.assert_allclose(result, expected)
    
    def test_parse_array_invalid(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_array("5,abc,15,20")

class TestLs2DvInterface(unittest.TestCase):
    def setUp(self):
        self.original_argv = sys.argv.copy()
        self.original_stdout = sys.stdout
        self.captured_output = StringIO()
        sys.stdout = self.captured_output
        self.original_show = plt.show
        plt.show = lambda: None

    def tearDown(self):
        sys.argv = self.original_argv
        sys.stdout = self.original_stdout
        plt.show = self.original_show

    def test_cli_valid_1D(self):
        sys.argv = [
            "ls_2Dv_interface.py",
            "--b", "1",
            "--f", "5",
            "--c", "1500",
            "--e", "0",
            "--x", "0",
            "--z", "5,200,500",
            "--plot-mode", "1D",
            "--N", "20"  # Using 20 segments
        ]
        try:
            cli_main()
        except SystemExit:
            self.fail("cli_main() raised SystemExit unexpectedly in 1D mode!")
        output = self.captured_output.getvalue()
        self.assertEqual(output, "")

    def test_cli_valid_2D(self):
        sys.argv = [
            "ls_2Dv_interface.py",
            "--b", "1",
            "--f", "5",
            "--c", "1500",
            "--e", "0",
            "--x2=-10,10,200",
            "--z2=1,20,200",
            "--plot-mode", "2D",
            "--N", "20"
        ]
        try:
            cli_main()
        except SystemExit:
            self.fail("cli_main() raised SystemExit unexpectedly in 2D mode!")
        output = self.captured_output.getvalue()
        self.assertEqual(output, "")

    def test_cli_valid_both(self):
        sys.argv = [
            "ls_2Dv_interface.py",
            "--b", "1",
            "--f", "5",
            "--c", "1500",
            "--e", "0",
            "--x", "0",
            "--z", "5,200,500",
            "--x2=-10,10,200",
            "--z2=1,20,200",
            "--plot-mode", "both",
            "--N", "20"
        ]
        try:
            cli_main()
        except SystemExit:
            self.fail("cli_main() raised SystemExit unexpectedly in both mode!")
        output = self.captured_output.getvalue()
        self.assertEqual(output, "")

    def test_cli_invalid_b(self):
        sys.argv = [
            "ls_2Dv_interface.py",
            "--b", "invalid_expr",
            "--f", "5",
            "--c", "1500",
            "--e", "0",
            "--x", "0"
        ]
        with self.assertRaises(SystemExit):
            cli_main()

    def test_cli_invalid_x2(self):
        sys.argv = [
            "ls_2Dv_interface.py",
            "--b", "1",
            "--f", "5",
            "--c", "1500",
            "--e", "0",
            "--x2=abc,10,200",
            "--z2=1,20,200",
            "--plot-mode", "2D"
        ]
        with self.assertRaises(SystemExit):
            cli_main()

    def test_cli_invalid_z2(self):
        sys.argv = [
            "ls_2Dv_interface.py",
            "--b", "1",
            "--f", "5",
            "--c", "1500",
            "--e", "0",
            "--x2=-10,10,200",
            "--z2=1,xyz,200",
            "--plot-mode", "2D"
        ]
        with self.assertRaises(SystemExit):
            cli_main()

class TestLs2DvService(unittest.TestCase):
    def test_service_returns_valid_output_1D(self):
        b = 1
        f = 5
        c = 1500
        e = 0
        x = 0
        z = np.linspace(5, 200, 500)
        p = run_ls_2Dv_service(b, f, c, e, x, z, N=20)
        self.assertTrue(isinstance(p, np.ndarray))
        self.assertEqual(np.prod(p.shape), z.size)

if __name__ == '__main__':
    unittest.main()
