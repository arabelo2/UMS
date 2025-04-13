# domain/init_xi3D.py

import numpy as np

class InitXi3D:
    """
    Class to determine the dimensions of the output array xi based on input coordinates (x, y, z)
    and to provide an empty array xi with dimensions (P, Q) as well as the dimension values.

    If the inputs do not follow the expected combinations, a ValueError is raised.
    
    Note:
      - For standard 2D evaluation grids (where each input is a 2D array, row/column vectors, or scalars),
        the original logic is preserved.
      - If any input has more than 2 dimensions (e.g. a 3D evaluation grid produced by np.meshgrid with three vectors),
        the inputs are flattened and the output xi is returned as a column vector with shape (N, 1), where
        N is the total number of evaluation points.
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
        # Note: If the input already has ndim > 2, it will be preserved.
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
        
        New behavior for multi-dimensional inputs:
          - If any of the input arrays has ndim > 2, the arrays are flattened to 1D, and xi is
            returned as an array of shape (N, 1), where N is the total number of evaluation points.
          - This “collapses” the extra dimensions while preserving the total number of points.
        """
        # If inputs are strictly 2D (the expected case)
        if self.x.ndim == 2 and self.y.ndim == 2 and self.z.ndim == 2:
            nrx, ncx = self.x.shape
            nry, ncy = self.y.shape
            nrz, ncz = self.z.shape

            # Check for supported combinations
            # New case: Full 2D matrices (2,2)
            if nrx == nry == nrz and ncx == ncy == ncz:
                P, Q = nrx, ncx
            # Case 1: x and z are matrices (same size), y is scalar
            elif nrx == nrz and ncx == ncz and nry == 1 and ncy == 1:
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
            # Case 12: x and z are row vectors, y is scalar (different column sizes)
            elif nrx == 1 and nrz == 1 and nry == 1 and ncy == 1:
                P, Q = 1, max(ncx, ncz)  # Use the larger of ncx or ncz
            else:
                raise ValueError(f"Unsupported (x, y, z) combination: shapes {self.x.shape}, {self.y.shape}, {self.z.shape}")

            return np.zeros((P, Q)), P, Q

        # New feature: Handle inputs with more than 2 dimensions (e.g. a full 3D grid)
        elif self.x.ndim > 2 or self.y.ndim > 2 or self.z.ndim > 2:
            # Capture the original shape and total number of points.
            orig_shape = self.x.shape  # Expecting all three have the same shape.
            N = self.x.size

            # Flatten the arrays to 1D.
            flat_x = self.x.flatten()
            flat_y = self.y.flatten()
            flat_z = self.z.flatten()

            # Create an output array for xi with one value per evaluation point.
            flat_xi = np.zeros(N)
            # (Note: The actual computation of intersection is done later in a loop
            # in the calling routine, so here we just provide a zero array.)

            # Return the flattened result reshaped as a column vector.
            # This effectively collapses the extra dimensions.
            return flat_xi.reshape((N, 1)), N, 1
        else:
            raise ValueError("Unexpected array dimensionality")

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
