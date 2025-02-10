import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
from application.ls_2Dint_service import run_ls_2Dint_service

class TestLs2Dint(unittest.TestCase):
    def test_default_parameters(self):
        """
        Test ls_2Dint with default parameters:
          b=3, f=5, c=1500, e=0,
          mat=[1,1480,7.9,5900],
          angt=45, Dt0=50,
          x=0, z=linspace(5,80,200)
        """
        b = 3
        f = 5
        c = 1500
        e = 0
        mat = [1, 1480, 7.9, 5900]
        angt = 45
        Dt0 = 50
        x = 0
        z = np.linspace(5, 80, 200)
        p = run_ls_2Dint_service(b, f, c, e, mat, angt, Dt0, x, z)
        self.assertTrue(isinstance(p, np.ndarray))
        self.assertEqual(p.shape, z.shape)
        self.assertTrue(np.iscomplexobj(p))
    
    def test_with_specified_segments(self):
        """
        Test ls_2Dint with an explicit segment count.
        """
        b = 3
        f = 5
        c = 1500
        e = 0
        mat = [1, 1480, 7.9, 5900]
        angt = 45
        Dt0 = 50
        x = 0
        z = np.linspace(5, 80, 200)
        N = 20
        p = run_ls_2Dint_service(b, f, c, e, mat, angt, Dt0, x, z, N)
        self.assertTrue(isinstance(p, np.ndarray))
        self.assertEqual(p.shape, z.shape)
    
    def test_invalid_material_vector(self):
        """
        Test that an invalid material vector (not four elements) raises an error.
        """
        with self.assertRaises(Exception):
            run_ls_2Dint_service(3, 5, 1500, 0, [1, 1480, 7.9], 45, 50, 0, np.linspace(5, 80, 200))
    
    def test_invalid_z_format(self):
        """
        Test that an invalid format for z (non-numeric string) raises an error.
        """
        with self.assertRaises(Exception):
            run_ls_2Dint_service(3, 5, 1500, 0, [1,1480,7.9,5900], 45, 50, 0, "invalid")
    
    def test_edge_case_angles(self):
        """
        Test ls_2Dint with extreme angle values (e.g., 0 and 90 degrees) for stability.
        """
        b = 3
        f = 5
        c = 1500
        e = 0
        mat = [1,1480,7.9,5900]
        Dt0 = 50
        x = 0
        z = np.linspace(5, 80, 200)
        # Angle 0 degrees.
        p0 = run_ls_2Dint_service(b, f, c, e, mat, 0, Dt0, x, z)
        self.assertTrue(isinstance(p0, np.ndarray))
        # Angle 90 degrees.
        p90 = run_ls_2Dint_service(b, f, c, e, mat, 90, Dt0, x, z)
        self.assertTrue(isinstance(p90, np.ndarray))
    
    def test_non_numeric_input(self):
        """
        Test that non-numeric input for a parameter (e.g., b) raises an error.
        """
        with self.assertRaises(Exception):
            run_ls_2Dint_service("non-numeric", 5, 1500, 0, [1,1480,7.9,5900], 45, 50, 0, np.linspace(5,80,200))

if __name__ == '__main__':
    unittest.main()
