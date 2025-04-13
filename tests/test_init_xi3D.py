#!/usr/bin/env python3
# tests/test_init_xi3D.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import unittest
import numpy as np
from domain.init_xi3D import InitXi3D

class TestInitXi3D(unittest.TestCase):

    def test_equal_sized_matrices_xz_and_scalar_y(self):
        x = np.array([[1, 2], [3, 4]])
        y = 0
        z = np.array([[5, 6], [7, 8]])
        init_obj = InitXi3D(x, y, z)
        xi, P, Q = init_obj.create_zero_array()
        self.assertEqual(xi.shape, (2, 2))
        self.assertEqual(P, 2)
        self.assertEqual(Q, 2)

    def test_equal_sized_matrices_xy_and_scalar_z(self):
        x = np.array([[1, 2], [3, 4]])
        y = np.array([[5, 6], [7, 8]])
        z = 10
        init_obj = InitXi3D(x, y, z)
        xi, P, Q = init_obj.create_zero_array()
        self.assertEqual(xi.shape, (2, 2))
        self.assertEqual(P, 2)
        self.assertEqual(Q, 2)

    def test_equal_sized_matrices_yz_and_scalar_x(self):
        x = 0
        y = np.array([[1, 2], [3, 4]])
        z = np.array([[5, 6], [7, 8]])
        init_obj = InitXi3D(x, y, z)
        xi, P, Q = init_obj.create_zero_array()
        self.assertEqual(xi.shape, (2, 2))
        self.assertEqual(P, 2)
        self.assertEqual(Q, 2)

    def test_fully_2D_matrices_x_y_z(self):
        x = np.array([[1, 2], [3, 4]])
        y = np.array([[5, 6], [7, 8]])
        z = np.array([[9, 10], [11, 12]])
        init_obj = InitXi3D(x, y, z)
        xi, P, Q = init_obj.create_zero_array()
        self.assertEqual(xi.shape, (2, 2))
        self.assertEqual(P, 2)
        self.assertEqual(Q, 2)

    def test_z_row_vector_x_y_scalars(self):
        x = 1
        y = 2
        z = [5, 6, 7]  # row vector
        init_obj = InitXi3D(x, y, z)
        xi, P, Q = init_obj.create_zero_array()
        self.assertEqual(xi.shape, (1, 3))
        self.assertEqual(P, 1)
        self.assertEqual(Q, 3)

    def test_z_column_vector_x_y_scalars(self):
        x = 1
        y = 2
        z = [[5], [6], [7]]  # column vector
        init_obj = InitXi3D(x, y, z)
        xi, P, Q = init_obj.create_zero_array()
        self.assertEqual(xi.shape, (3, 1))
        self.assertEqual(P, 3)
        self.assertEqual(Q, 1)

    def test_x_row_vector_z_scalar_y_scalar(self):
        x = [1, 2, 3]  # row vector
        y = 4
        z = 5
        init_obj = InitXi3D(x, y, z)
        xi, P, Q = init_obj.create_zero_array()
        self.assertEqual(xi.shape, (1, 3))
        self.assertEqual(P, 1)
        self.assertEqual(Q, 3)

    def test_x_column_vector_z_scalar_y_scalar(self):
        x = [[1], [2], [3]]  # column vector
        y = 4
        z = 5
        init_obj = InitXi3D(x, y, z)
        xi, P, Q = init_obj.create_zero_array()
        self.assertEqual(xi.shape, (3, 1))
        self.assertEqual(P, 3)
        self.assertEqual(Q, 1)

    def test_y_row_vector_x_scalar_z_scalar(self):
        x = 1
        y = [2, 3, 4]  # row vector
        z = 5
        init_obj = InitXi3D(x, y, z)
        xi, P, Q = init_obj.create_zero_array()
        self.assertEqual(xi.shape, (1, 3))
        self.assertEqual(P, 1)
        self.assertEqual(Q, 3)

    def test_scalar_inputs(self):
        """Test when x, y, z are all scalars (should return a 1x1 zero matrix)."""
        x = 1
        y = 2
        z = 3
        init_obj = InitXi3D(x, y, z)
        xi, P, Q = init_obj.create_zero_array()
        self.assertEqual(xi.shape, (1, 1))
        self.assertEqual(P, 1)
        self.assertEqual(Q, 1)

    def test_empty_inputs(self):
        """Test when an empty list is passed (should raise ValueError)."""
        x = []
        y = 1
        z = 2
        with self.assertRaises(ValueError):
            InitXi3D(x, y, z)

    def test_non_numeric_input(self):
        """Test with non-numeric input (should raise ValueError)."""
        x = "a"
        y = 2
        z = 3
        with self.assertRaises(ValueError):
            InitXi3D(x, y, z)
    def test_3d_input_flattening(self):
        # Create a full 3D grid using np.meshgrid for x, y, and z.
        x = np.linspace(1, 3, 3)  # For example: 3 points
        y = np.linspace(4, 6, 3)  # 3 points
        z = np.linspace(7, 9, 3)  # 3 points
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        init_obj = InitXi3D(X, Y, Z)
        xi, P, Q = init_obj.create_zero_array()
        # Expect N = 3 * 3 * 3 = 27 evaluation points, so xi should have shape (27, 1)
        self.assertEqual(xi.shape, (27, 1))
        self.assertEqual(P, 27)
        self.assertEqual(Q, 1)

if __name__ == '__main__':
    unittest.main()
