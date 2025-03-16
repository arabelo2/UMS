# tests/test_init_xi.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import unittest
import numpy as np
from domain.init_xi import init_xi

class TestInitXiEdgeCases(unittest.TestCase):
    def test_scalar_scalar(self):
        # Both x and z are scalars.
        x = 5
        z = 10
        xi, P, Q = init_xi(x, z)
        self.assertEqual(xi.shape, (1, 1))
        self.assertEqual(P, 1)
        self.assertEqual(Q, 1)

    def test_list_vector(self):
        # x and z provided as lists; these become row vectors.
        x = [1, 2, 3]
        z = [4, 5, 6]
        xi, P, Q = init_xi(x, z)
        self.assertEqual(xi.shape, (1, 3))
        self.assertEqual(P, 1)
        self.assertEqual(Q, 3)
    
    def test_numpy_array_1d(self):
        # x and z as 1D numpy arrays (treated as row vectors).
        x = np.array([1, 2, 3])
        z = np.array([4, 5, 6])
        xi, P, Q = init_xi(x, z)
        self.assertEqual(xi.shape, (1, 3))
        self.assertEqual(P, 1)
        self.assertEqual(Q, 3)

    def test_x_column_vector_z_scalar(self):
        # x as a column vector and z as a scalar.
        x = np.array([[1], [2], [3]])  # 3x1 array
        z = 10  # scalar
        xi, P, Q = init_xi(x, z)
        self.assertEqual(xi.shape, (3, 1))
        self.assertEqual(P, 3)
        self.assertEqual(Q, 1)
    
    def test_z_column_vector_x_scalar(self):
        # z as a column vector and x as a scalar.
        z = np.array([[4], [5], [6]])  # 3x1 array
        x = 7  # scalar
        xi, P, Q = init_xi(x, z)
        self.assertEqual(xi.shape, (3, 1))
        self.assertEqual(P, 3)
        self.assertEqual(Q, 1)
    
    def test_x_row_vector_z_scalar(self):
        # x as a row vector and z as a scalar.
        x = np.array([[1, 2, 3]])  # 1x3 array
        z = 10  # scalar
        xi, P, Q = init_xi(x, z)
        self.assertEqual(xi.shape, (1, 3))
        self.assertEqual(P, 1)
        self.assertEqual(Q, 3)
    
    def test_z_row_vector_x_scalar(self):
        # z as a row vector and x as a scalar.
        z = np.array([[4, 5, 6]])  # 1x3 array
        x = 7  # scalar
        xi, P, Q = init_xi(x, z)
        self.assertEqual(xi.shape, (1, 3))
        self.assertEqual(P, 1)
        self.assertEqual(Q, 3)

    def test_equal_sized_matrices(self):
        # x and z as equal sized matrices.
        x = np.array([[1, 2], [3, 4]])
        z = np.array([[5, 6], [7, 8]])
        xi, P, Q = init_xi(x, z)
        self.assertEqual(xi.shape, (2, 2))
        self.assertEqual(P, 2)
        self.assertEqual(Q, 2)
    
    def test_invalid_mismatched_shapes(self):
        # x as a 2x2 matrix and z as a 1x2 row vector should raise an error.
        x = np.array([[1, 2], [3, 4]])
        z = [5, 6]  # becomes a 1x2 row vector.
        with self.assertRaises(ValueError):
            init_xi(x, z)
    
    def test_high_dimensional_invalid(self):
        # x is a 2x3 matrix and z is a 3x2 matrix; these shapes don't match.
        x = np.array([[1, 2, 3],
                      [4, 5, 6]])
        z = np.array([[7, 8],
                      [9, 10],
                      [11, 12]])
        with self.assertRaises(ValueError):
            init_xi(x, z)
    
    def test_empty_input(self):
        # Test with empty inputs should now raise a ValueError.
        x = []
        z = []
        with self.assertRaises(ValueError):
            init_xi(x, z)
    
    def test_non_numeric_input(self):
        # Test that non-numeric input raises a ValueError.
        with self.assertRaises(ValueError):
            init_xi("a string", 5)

if __name__ == '__main__':
    unittest.main()
