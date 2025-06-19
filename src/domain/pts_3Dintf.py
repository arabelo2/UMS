# domain/pts_3Dintf.py

import numpy as np
from domain.ferrari2 import ferrari2
from domain.init_xi3D import InitXi3D

class Pts3DIntf:
    """
    Class to compute the intersection distance xi along the interface for a given
    array element and observation point in a second medium.

    Attributes:
        ex   : float - Offset of the element center from the array center (mm).
        ey   : float - Offset of the element center from the array center along y (mm).
        xn   : float - Offset of the segment from the element center in x (mm).
        yn   : float - Offset of the segment from the element center in y (mm).
        angt : float - Angle of the array relative to the interface (degrees).
        Dt0  : float - Distance from the array center to the interface (mm).
        c1   : float - Wave speed in medium one (m/s).
        c2   : float - Wave speed in medium two (m/s).
    """

    def __init__(self, ex, ey, xn, yn, angt, Dt0, c1, c2):
        self.ex = ex
        self.ey = ey
        self.xn = xn
        self.yn = yn
        self.angt = np.deg2rad(angt)  # Convert to radians for computation
        self.Dt0 = Dt0
        self.c1 = c1
        self.c2 = c2
        self.cr = c1 / c2  # Wave speed ratio

    def compute_intersection(self, x, y, z):
        """
        Compute the intersection distance xi, handling all combinations of scalar/vector inputs.
        """
        # Convert inputs to numpy arrays
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        z_arr = np.asarray(z)
        
        # Determine input shapes and dimensionality
        x_scalar = x_arr.ndim == 0
        y_scalar = y_arr.ndim == 0
        z_scalar = z_arr.ndim == 0
        
        # Get broadcast shape for all inputs
        try:
            # This will raise ValueError if shapes are incompatible
            broadcast_shape = np.broadcast_shapes(
                np.shape(x),
                np.shape(y),
                np.shape(z)
            )
        except ValueError as e:
            raise ValueError(f"Input shapes incompatible for broadcasting: {np.shape(x)}, {np.shape(y)}, {np.shape(z)}") from e
        
        # Broadcast all inputs to common shape
        x_bc = np.broadcast_to(x_arr, broadcast_shape)
        y_bc = np.broadcast_to(y_arr, broadcast_shape)
        z_bc = np.broadcast_to(z_arr, broadcast_shape)
        
        # Initialize xi array with broadcast shape
        xi = np.zeros(broadcast_shape)
        
        # Compute initial xi using the InitXi3D helper
        init_xi = InitXi3D(x_bc.ravel(), y_bc.ravel(), z_bc.ravel())
        xi_flat, P, Q = init_xi.compute()
        xi = xi_flat.reshape(broadcast_shape)
        
        # Compute effective distances:
        De = self.Dt0 + (self.ex + self.xn) * np.sin(self.angt)  # scalar
        
        # Process z values safely (avoid division by zero)
        z_safe = np.where(z_bc <= 0, 1e-6, z_bc)
        
        # Compute Dx and Dy with broadcasting
        Dx = x_bc - (self.ex + self.xn) * np.cos(self.angt)
        Dy = y_bc - (self.ey + self.yn)
        Db = np.sqrt(Dx**2 + Dy**2)
        
        # Iterate over each evaluation point and compute xi
        for idx in np.ndindex(*broadcast_shape):
            xi[idx] = ferrari2(self.cr, z_safe[idx], De, Db[idx])
        
        # Return with original scalar/vector shape
        if x_scalar and y_scalar and z_scalar:
            return xi.item()  # return as scalar if all inputs were scalar
        return xi
