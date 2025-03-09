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
        Compute the intersection distance xi where the ray from a segment of the array element
        intersects the interface.

        Parameters:
            x : scalar, list, or numpy array - x-coordinates of the observation point (mm).
            y : scalar, list, or numpy array - y-coordinates of the observation point (mm).
            z : scalar, list, or numpy array - z-coordinates of the observation point (mm).

        Returns:
            xi : numpy array - The intersection distances along the interface (mm).
        """
        # Initialize the output matrix using InitXi3D
        init_xi = InitXi3D(x, y, z)
        xi, P, Q = init_xi.compute()

        # Compute effective distances
        De = self.Dt0 + (self.ex + self.xn) * np.sin(self.angt)  # Effective height
        Dx = x - (self.ex + self.xn) * np.cos(self.angt)         # Horizontal distance
        Dy = y - (self.ey + self.yn)                             # Lateral distance
        Db = np.sqrt(Dx**2 + Dy**2)                              # Total in-plane distance

        # Iterate through the matrix dimensions to compute xi values
        for pp in range(P):
            for qq in range(Q):
                if np.isscalar(Db):
                    xi[pp, qq] = ferrari2(self.cr, z, De, Db)
                else:
                    xi[pp, qq] = ferrari2(self.cr, z[pp, qq], De, Db[pp, qq])

        return xi
