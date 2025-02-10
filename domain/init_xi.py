# domain/init_xi.py

import numpy as np

def init_xi(x, z):
    """
    Determine the dimensions for the output array xi based on the inputs x and z.
    
    Parameters:
        x: scalar, list, or numpy array representing x coordinates.
        z: scalar, list, or numpy array representing z coordinates.
        
    Returns:
        xi: numpy array of zeros with dimensions (P, Q)
        P: int, number of rows
        Q: int, number of columns
        
    The dimensions are determined as follows:
      - If x and z have the same shape, then xi is an array of zeros with that shape.
      - If x is a column vector (n×1) and z is a scalar (1×1), then xi is an (n×1) column vector.
      - If z is a column vector (n×1) and x is a scalar (1×1), then xi is an (n×1) column vector.
      - If x is a row vector (1×n) and z is a scalar (1×1), then xi is a (1×n) row vector.
      - If z is a row vector (1×n) and x is a scalar (1×1), then xi is a (1×n) row vector.
      - Otherwise, a ValueError is raised.
    """
    # Attempt to convert inputs to numpy arrays with float conversion.
    try:
        x_arr = np.array(x, dtype=float, ndmin=2)
        z_arr = np.array(z, dtype=float, ndmin=2)
    except Exception as e:
        raise ValueError("Inputs must be numeric") from e

    # Verify that the arrays have a numeric dtype.
    if not np.issubdtype(x_arr.dtype, np.number) or not np.issubdtype(z_arr.dtype, np.number):
        raise ValueError("Inputs must be numeric")

    # Explicitly check for empty dimensions.
    if x_arr.shape[0] == 0 or x_arr.shape[1] == 0 or z_arr.shape[0] == 0 or z_arr.shape[1] == 0:
        raise ValueError("Empty input is not allowed")
    
    # Get shapes.
    nrx, ncx = x_arr.shape
    nrz, ncz = z_arr.shape
    
    # Case 1: x and z have the same shape.
    if (nrx == nrz) and (ncx == ncz):
        xi = np.zeros((nrx, ncx))
        P = nrx
        Q = ncx
        return xi, P, Q
    # Case 2: x is a column vector and z is a scalar.
    elif (nrx > 1) and (ncx == 1) and (nrz == 1) and (ncz == 1):
        xi = np.zeros((nrx, 1))
        P = nrx
        Q = 1
        return xi, P, Q
    # Case 3: z is a column vector and x is a scalar.
    elif (nrz > 1) and (ncz == 1) and (nrx == 1) and (ncx == 1):
        xi = np.zeros((nrz, 1))
        P = nrz
        Q = 1
        return xi, P, Q
    # Case 4: x is a row vector and z is a scalar.
    elif (ncx > 1) and (nrx == 1) and (nrz == 1) and (ncz == 1):
        xi = np.zeros((1, ncx))
        P = 1
        Q = ncx
        return xi, P, Q
    # Case 5: z is a row vector and x is a scalar.
    elif (ncz > 1) and (nrz == 1) and (nrx == 1) and (ncx == 1):
        xi = np.zeros((1, ncz))
        P = 1
        Q = ncz
        return xi, P, Q
    else:
        raise ValueError("(x,z) must be (vector,scalar) pairs or equal matrices")
