# domain/init_xi3D.py

# domain/init_xi3D.py

import numpy as np

class InitXi3D:
    """
    Class to determine the dimensions of the output array xi based on input coordinates (x, y, z)
    and to provide an empty array xi with dimensions (P, Q) as well as the dimension values.

    If the inputs do not follow the expected combinations, a ValueError is raised.
    """

    def __init__(self, x, y, z):
        """
        Initialize the InitXi3D object with input coordinates.

        Parameters:
            x: scalar, list, or numpy array representing x-coordinates.
            y: scalar, list, or numpy array representing y-coordinates.
            z: scalar, list, or numpy array representing z-coordinates.
        """
        self.x = self._to_array(x)
        self.y = self._to_array(y)
        self.z = self._to_array(z)

        # Ensure inputs are at least 2D for consistent shape comparisons.
        self.x = np.atleast_2d(self.x)
        self.y = np.atleast_2d(self.y)
        self.z = np.atleast_2d(self.z)

    def _to_array(self, value):
        """
        Converts input to a NumPy array, ensuring numeric type.
        """
        try:
            arr = np.array(value, dtype=float)
            if arr.size == 0:
                raise ValueError("Input arrays cannot be empty")
            return arr
        except Exception as e:
            raise ValueError("Inputs must be numeric") from e

    def compute(self):
        """
        Compute the output array xi (filled with zeros) and the dimensions (P, Q) based on the
        shapes of x, y, and z.

        Returns:
            xi: numpy array of zeros with shape (P, Q)
            P: int, number of rows
            Q: int, number of columns

        Raises:
            ValueError: If the combination of input shapes is not supported.
        """
        nrx, ncx = self.x.shape
        nry, ncy = self.y.shape
        nrz, ncz = self.z.shape

        # Check for supported combinations
        # Case 1: x and z are matrices (same size), y is scalar
        if nrx == nrz and ncx == ncz and nry == 1 and ncy == 1:
            P, Q = nrx, ncx
        # Case 2: x and y are matrices (same size), z is scalar
        elif nrx == nry and ncx == ncy and nrz == 1 and ncz == 1:
            P, Q = nrx, ncx
        # Case 3: y and z are matrices (same size), x is scalar
        elif nry == nrz and ncy == ncz and nrx == 1 and ncx == 1:
            P, Q = nry, ncy
        # Case 4: z is a row vector, x and y are scalars
        elif nrz == 1 and ncz > 1 and nrx == 1 and ncx == 1 and nry == 1 and ncy == 1:
            P, Q = 1, ncz
        # Case 5: z is a column vector, x and y are scalars
        elif ncz == 1 and nrz > 1 and nrx == 1 and ncx == 1 and nry == 1 and ncy == 1:
            P, Q = nrz, 1
        # Case 6: x is a row vector, y and z are scalars
        elif ncx > 1 and nrx == 1 and nry == 1 and ncy == 1 and nrz == 1 and ncz == 1:
            P, Q = 1, ncx
        # Case 7: x is a column vector, y and z are scalars
        elif nrx > 1 and ncx == 1 and nry == 1 and ncy == 1 and nrz == 1 and ncz == 1:
            P, Q = nrx, 1
        # Case 8: y is a row vector, x and z are scalars
        elif ncy > 1 and nry == 1 and nrx == 1 and ncx == 1 and nrz == 1 and ncz == 1:
            P, Q = 1, ncy
        # Case 9: y is a column vector, x and z are scalars
        elif nry > 1 and ncy == 1 and nrx == 1 and ncx == 1 and nrz == 1 and ncz == 1:
            P, Q = nry, 1
        # Case 10: x, y, z are row vectors (same size)
        elif nrx == 1 and nry == 1 and nrz == 1 and ncx == ncy and ncx == ncz:
            P, Q = 1, ncx
        # Case 11: x, y, z are column vectors (same size)
        elif ncx == 1 and ncy == 1 and ncz == 1 and nrx == nry and nrx == nrz:
            P, Q = nrx, 1
        else:
            # If no valid case matches, raise an error.
            raise ValueError(f"Unsupported (x, y, z) combination: shapes {self.x.shape}, {self.y.shape}, {self.z.shape}")

        return np.zeros((P, Q)), P, Q

    def create_zero_array(self):
        """
        Alias for `compute()` to maintain compatibility with existing test cases.
        """
        return self.compute()

if __name__ == '__main__':
    # For simple manual testing.
    # Example: x, y as scalars and z as a row vector.
    xi_instance = InitXi3D(1, 2, [5, 6, 7])
    xi, P, Q = xi_instance.create_zero_array()
    print("xi shape:", xi.shape, "P:", P, "Q:", Q)
