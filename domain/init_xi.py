# domain/init_xi.py

"""
Module: init_xi.py
Layer: Domain

Provides the core logic for initializing the xi array based on the inputs x and z.
The supported configurations are:
  - x and z are equal-sized (scalars, vectors, or matrices),
  - x is a vector (row or column) and z is a scalar,
  - z is a vector (row or column) and x is a scalar.
Any other combination raises a ValueError.
"""

from typing import Tuple, Any
import numpy as np


class InitXiSolver:
    """
    Solver for initializing an empty array xi and determining its dimensions (P, Q)
    from the input coordinates x and z.
    """

    def __init__(self, x: Any, z: Any) -> None:
        """
        Initialize the solver with x and z.

        Parameters:
            x: A scalar, vector, or matrix representing x-coordinates.
            z: A scalar, vector, or matrix representing z-coordinates.
        """
        self.x = self._to_2d(x)
        self.z = self._to_2d(z)

    @staticmethod
    def _to_2d(a: Any) -> np.ndarray:
        """
        Convert input to a 2-D NumPy array.

        - Scalars are reshaped to (1,1)
        - 1-D arrays are treated as row vectors (1 x n)

        Parameters:
            a: Input value (scalar, list, or array)
        Returns:
            A 2-D NumPy array.
        """
        arr = np.array(a)
        if arr.ndim == 0:
            return arr.reshape(1, 1)
        elif arr.ndim == 1:
            return arr.reshape(1, -1)
        return arr

    def compute(self) -> Tuple[np.ndarray, int, int]:
        """
        Compute the output array xi and its dimensions (P, Q) based on x and z.

        Returns:
            xi (np.ndarray): An array of zeros with dimensions (P, Q).
            P (int): The number of rows.
            Q (int): The number of columns.

        Raises:
            ValueError: If (x, z) do not match one of the allowed configurations.
        """
        x_arr = self.x
        z_arr = self.z
        nrx, ncx = x_arr.shape
        nrz, ncz = z_arr.shape

        # Case 1: Equal-sized arrays (scalars, vectors, or matrices).
        if nrx == nrz and ncx == ncz:
            xi = np.zeros((nrx, ncx))
            P, Q = nrx, ncx

        # Case 2: x is a column vector and z is a scalar.
        elif (nrx > 1 and ncx == 1) and (nrz == 1 and ncz == 1):
            xi = np.zeros((nrx, 1))
            P, Q = nrx, 1

        # Case 3: z is a column vector and x is a scalar.
        elif (nrz > 1 and ncz == 1) and (nrx == 1 and ncx == 1):
            xi = np.zeros((nrz, 1))
            P, Q = nrz, 1

        # Case 4: x is a row vector and z is a scalar.
        elif (ncx > 1 and nrx == 1) and (nrz == 1 and ncz == 1):
            xi = np.zeros((1, ncx))
            P, Q = 1, ncx

        # Case 5: z is a row vector and x is a scalar.
        elif (ncz > 1 and nrz == 1) and (nrx == 1 and ncx == 1):
            xi = np.zeros((1, ncz))
            P, Q = 1, ncz

        else:
            raise ValueError("(x, z) must be (vector, scalar) pairs or equal matrices")

        return xi, P, Q
