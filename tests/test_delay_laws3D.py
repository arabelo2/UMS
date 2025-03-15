# tests/test_delay_laws3D.py

import unittest
import subprocess
import os

def run_cli(args):
    """
    Helper function that calls delay_laws3D_interface.py with provided args.
    Returns stdout, stderr, and the exit code.
    """
    cmd = ["python", "interface/delay_laws3D_interface.py"] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout, result.stderr, result.returncode

class TestDelayLaws3DInterface(unittest.TestCase):
    def setUp(self):
        """
        Remove default output file if it exists before each test.
        """
        self.outfile = "delay_laws3D_output.txt"
        if os.path.exists(self.outfile):
            os.remove(self.outfile)
    
    def test_default_run(self):
        """
        Test running with default parameters.
        Expect exit code 0 and output file created.
        """
        stdout, stderr, code = run_cli([])
        self.assertEqual(code, 0, f"CLI exited with code {code}. stderr: {stderr}")
        self.assertIn("Time delays saved to", stdout)
        self.assertTrue(os.path.exists(self.outfile))
    
    def test_focusing_mode(self):
        """
        Provide a focusing scenario by setting F to a finite value.
        """
        stdout, stderr, code = run_cli(["--F", "11", "--plot", "N"])
        self.assertEqual(code, 0)
        self.assertIn("Time delays saved to", stdout)
        self.assertTrue(os.path.exists(self.outfile))
    
    def test_invalid_M_zero(self):
        """
        M=0 should raise a ValueError.
        """
        stdout, stderr, code = run_cli(["--M", "0"])
        self.assertNotEqual(code, 0)
        self.assertIn("M", stderr.lower())
    
    def test_invalid_wave_speed_zero(self):
        """
        c=0 should raise a ValueError regarding division by zero.
        """
        stdout, stderr, code = run_cli(["--c", "0"])
        self.assertNotEqual(code, 0)
        self.assertIn("zero", stderr.lower())
    
    def test_custom_parameters(self):
        """
        Test a run with custom parameters.
        """
        stdout, stderr, code = run_cli([
            "--M", "10", "--N", "12", "--sx", "0.2", "--sy", "0.25",
            "--theta", "30", "--phi", "15", "--F", "inf", "--c", "1500", "--plot", "N"
        ])
        self.assertEqual(code, 0)
        self.assertIn("Time delays saved to", stdout)
        self.assertTrue(os.path.exists(self.outfile))

if __name__ == '__main__':
    unittest.main()
