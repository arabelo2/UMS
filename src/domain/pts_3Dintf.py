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
        Compute the intersection distance xi.
        """
        # Convert inputs to arrays (ensuring at least 2D)
        x_arr = np.atleast_2d(x)
        y_arr = np.atleast_2d(y)
        z_arr = np.atleast_2d(z)

        # Check if any input is higher-dimensional (e.g., full 3D grid)
        if x_arr.ndim > 2 or y_arr.ndim > 2 or z_arr.ndim > 2:
            orig_shape = x_arr.shape  # e.g., (81, 810) from meshgrid of x and y
            # Flatten the arrays
            x_proc = x_arr.flatten()
            y_proc = y_arr.flatten()
            z_proc = z_arr.flatten()
        else:
            orig_shape = x_arr.shape
            x_proc = x_arr
            y_proc = y_arr
            z_proc = z_arr

        # Compute initial xi using the InitXi3D helper.
        init_xi = InitXi3D(x_proc, y_proc, z_proc)
        xi, P, Q = init_xi.compute()

        # If we flattened for a high-dimensional grid, reshape xi back to the original grid shape.
        if x_arr.ndim > 2 or y_arr.ndim > 2 or z_arr.ndim > 2:
            xi = xi.reshape(orig_shape)

        # Compute effective distances:
        De = self.Dt0 + (self.ex + self.xn) * np.sin(self.angt)  # scalar
        Dx = x_proc - (self.ex + self.xn) * np.cos(self.angt)
        Dy = y_proc - (self.ey + self.yn)
        Db_flat = np.sqrt(Dx**2 + Dy**2)
        Db = Db_flat.reshape(orig_shape)

        # Process z: avoid division by zero.
        z_safe_flat = np.where(np.array(z_proc) <= 0, 1e-6, np.array(z_proc))
        # If the constant z value was provided (size 1), broadcast it over the grid.
        if z_safe_flat.size == 1:
            z_safe = np.full(orig_shape, z_safe_flat[0])
        else:
            z_safe = z_safe_flat.reshape(orig_shape)

        # Iterate over each evaluation point and compute xi.
        for idx, _ in np.ndenumerate(xi):
            xi[idx] = ferrari2(self.cr, z_safe[idx], De, Db[idx])

        return xi.squeeze()
