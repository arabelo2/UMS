# tests/test_pts_2Dintf.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import sys
from io import StringIO
from unittest.mock import patch

class TestPts2DintfInterface(unittest.TestCase):
    def setUp(self):
        # Backup original sys.stdout and sys.stderr.
        self.orig_stdout = sys.stdout
        self.orig_stderr = sys.stderr
        self.out = StringIO()
        self.err = StringIO()
        sys.stdout = self.out
        sys.stderr = self.err

    def tearDown(self):
        sys.stdout = self.orig_stdout
        sys.stderr = self.orig_stderr

    def run_interface(self, args_list):
        """
        Helper method to simulate command-line execution of pts_2Dintf_interface.py.
        Returns a tuple (exit_code, stdout_content, stderr_content).
        """
        from interface.pts_2Dintf_interface import main
        with patch.object(sys, 'argv', args_list):
            try:
                main()
            except SystemExit as e:
                # Capture sys.exit() calls (e.g., from argparse.error).
                return e.code, self.out.getvalue(), self.err.getvalue()
        return 0, self.out.getvalue(), self.err.getvalue()

    def test_default_parameters(self):
        """
        Test running the CLI with no additional parameters (using defaults).
        """
        args = ["pts_2Dintf_interface.py"]
        exit_code, output, err_output = self.run_interface(args)
        self.assertEqual(exit_code, 0)
        self.assertIn("Computed intersection point xi:", output)

    def test_extreme_angle_zero(self):
        """
        Test with an extreme angle of 0 degrees.
        """
        args = ["pts_2Dintf_interface.py", "--angt", "0"]
        exit_code, output, err_output = self.run_interface(args)
        self.assertEqual(exit_code, 0)
        self.assertIn("Computed intersection point xi:", output)

    def test_extreme_angle_ninety(self):
        """
        Test with an extreme angle of 90 degrees.
        """
        args = ["pts_2Dintf_interface.py", "--angt", "90"]
        exit_code, output, err_output = self.run_interface(args)
        self.assertEqual(exit_code, 0)
        self.assertIn("Computed intersection point xi:", output)

    def test_negative_values(self):
        """
        Test with negative values for parameters such as e, xc, and x.
        """
        args = [
            "pts_2Dintf_interface.py",
            "--e", "-5",
            "--xc", "-1.5",
            "--Dt0", "100",
            "--c1", "1480",
            "--c2", "5900",
            "--x", "-50",
            "--z", "80"
        ]
        exit_code, output, err_output = self.run_interface(args)
        self.assertEqual(exit_code, 0)
        self.assertIn("Computed intersection point xi:", output)

    def test_custom_parameters(self):
        """
        Test the CLI with custom valid parameters.
        """
        args = [
            "pts_2Dintf_interface.py",
            "--e", "2",
            "--xc", "1.0",
            "--angt", "30",
            "--Dt0", "120",
            "--c1", "1500",
            "--c2", "6000",
            "--x", "40",
            "--z", "90"
        ]
        exit_code, output, err_output = self.run_interface(args)
        self.assertEqual(exit_code, 0)
        self.assertIn("Computed intersection point xi:", output)

    def test_invalid_numeric_input(self):
        """
        Test that non-numeric input for a parameter (e.g., Dt0) results in an error.
        """
        args = [
            "pts_2Dintf_interface.py",
            "--Dt0", "invalid"
        ]
        exit_code, output, err_output = self.run_interface(args)
        # We expect a nonzero exit code.
        self.assertNotEqual(exit_code, 0)
        # Combine stdout and stderr output.
        combined_output = output + err_output
        self.assertTrue("invalid" in combined_output.lower() or "error" in combined_output.lower())

if __name__ == '__main__':
    unittest.main()
