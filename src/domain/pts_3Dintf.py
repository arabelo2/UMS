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

        # Convert inputs to NumPy arrays
        x = np.atleast_2d(x)  # Ensures a minimum of 2D
        y = np.atleast_2d(y)
        z = np.atleast_2d(z)

        # Initialize intersection matrix
        init_xi = InitXi3D(x, y, z)
        xi, P, Q = init_xi.compute()

        # Compute effective distances
        De = self.Dt0 + (self.ex + self.xn) * np.sin(self.angt)  # Effective height
        Dx = x - (self.ex + self.xn) * np.cos(self.angt)         # Horizontal distance
        Dy = y - (self.ey + self.yn)                             # Lateral distance
        Db = np.sqrt(Dx**2 + Dy**2)                              # Total in-plane distance

        # Ensure `z` is positive
        z = np.where(z <= 0, 1e-6, z)  # Avoid zero depth issues

        # Iterate through the matrix dimensions
        for pp in range(P):
            for qq in range(Q):
                xi[pp, qq] = ferrari2(self.cr, z[pp, qq], De, Db[pp, qq])

        # Fix shape issues: remove unnecessary dimensions
        return xi.squeeze()
