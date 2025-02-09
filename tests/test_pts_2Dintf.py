# tests/test_pts_2Dintf.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
from application.pts_2Dintf_service import Pts2DIntfService

class TestPts2DIntf(unittest.TestCase):
    def test_scalar_x_vector_z(self):
        # x is scalar, z is a vector.
        e = 0.0
        xn = 0.0
        angt = 0.0
        Dt0 = 10.0
        c1 = 1500.0
        c2 = 1600.0  # different media: cr != 1
        x = 5.0
        z = [10.0, 12.0, 14.0]
        service = Pts2DIntfService(e, xn, angt, Dt0, c1, c2, x, z)
        xi = service.compute()
        # Expect xi to be a numpy array with as many elements as there are in z.
        self.assertTrue(isinstance(xi, np.ndarray))
        self.assertEqual(xi.shape, (3,))
    
    def test_vector_x_scalar_z(self):
        # x is a vector, z is scalar.
        e = 0.0
        xn = 0.0
        angt = 0.0
        Dt0 = 10.0
        c1 = 1500.0
        c2 = 1400.0  # different media
        x = [2.0, 4.0, 6.0]
        z = 15.0
        service = Pts2DIntfService(e, xn, angt, Dt0, c1, c2, x, z)
        xi = service.compute()
        # Expect xi to have the same shape as x when broadcast (i.e. (3,))
        self.assertTrue(isinstance(xi, np.ndarray))
        self.assertEqual(xi.shape, (3,))
    
    def test_matrix_inputs(self):
        # Both x and z are 2x2 matrices.
        e = 1.0
        xn = 0.5
        angt = 30.0
        Dt0 = 10.0
        c1 = 1500.0
        c2 = 1550.0
        x = np.array([[5.0, 6.0], [7.0, 8.0]])
        z = np.array([[10.0, 11.0], [12.0, 13.0]])
        service = Pts2DIntfService(e, xn, angt, Dt0, c1, c2, x, z)
        xi = service.compute()
        self.assertTrue(isinstance(xi, np.ndarray))
        self.assertEqual(xi.shape, (2, 2))

if __name__ == "__main__":
    unittest.main()
