"""
Module: test_init_xi.py
Layer: Tests

Provides unit tests for the init_xi functionality.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
from application.init_xi_service import InitXiService

class TestInitXi(unittest.TestCase):
    def test_scalar_scalar(self):
        # Both x and z are scalars.
        service = InitXiService(5, 10)
        xi, P, Q = service.compute()
        self.assertEqual((P, Q), (1, 1))
        self.assertEqual(xi.shape, (1, 1))
    
    def test_column_vector_x_scalar(self):
        # x is a column vector, z is scalar.
        x = [[1], [2], [3]]
        z = 10
        service = InitXiService(x, z)
        xi, P, Q = service.compute()
        self.assertEqual((P, Q), (3, 1))
        self.assertEqual(xi.shape, (3, 1))
    
    def test_row_vector_x_scalar(self):
        # x is a row vector, z is scalar.
        x = [1, 2, 3]
        z = 10
        service = InitXiService(x, z)
        xi, P, Q = service.compute()
        self.assertEqual((P, Q), (1, 3))
        self.assertEqual(xi.shape, (1, 3))
    
    def test_equal_matrices(self):
        # x and z are equal-sized matrices.
        x = [[1, 2], [3, 4]]
        z = [[5, 6], [7, 8]]
        service = InitXiService(x, z)
        xi, P, Q = service.compute()
        self.assertEqual((P, Q), (2, 2))
        self.assertEqual(xi.shape, (2, 2))
    
    def test_invalid_configuration(self):
        # Test that an invalid combination raises a ValueError.
        x = [1, 2, 3]  # row vector of shape (1,3)
        z = [[5], [6]]  # column vector of shape (2,1)
        with self.assertRaises(ValueError):
            InitXiService(x, z).compute()


if __name__ == '__main__':
    unittest.main()
