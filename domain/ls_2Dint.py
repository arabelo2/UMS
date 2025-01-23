import numpy as np
from domain.pts_2Dintf import Points2DInterface


class LS2DInterface:
    """
    Computes normalized pressure for an element in a 1-D array radiating waves
    across a plane fluid/fluid interface.
    """

    def __init__(self, b, f, mat, angt, Dt0):
        """
        Initialize the LS2DInterface object.

        Parameters:
            b (float): Half-length of the source (in mm).
            f (float): Frequency (in MHz).
            mat (list): Material properties [d1, c1, d2, c2].
            angt (float): Angle of the array with respect to the x-axis (in degrees).
            Dt0 (float): Distance of the center of the array from the interface (in mm).
        """
        self.b = b
        self.f = f
        self.d1, self.c1, self.d2, self.c2 = mat
        self.angt = angt
        self.Dt0 = Dt0
        self.k1b = 2000 * np.pi * b * f / self.c1
        self.k2b = 2000 * np.pi * b * f / self.c2

    def compute_pressure(self, x, z, e, N=None):
        """
        Calculate the normalized pressure at a point (x, z).

        Parameters:
            x (float): x-coordinate of the evaluation point (in mm).
            z (float): z-coordinate of the evaluation point (in mm).
            e (float): Offset of the center of the element from the center of the array (in mm).
            N (int, optional): Number of segments. If not provided, it is calculated automatically.

        Returns:
            complex: Normalized pressure at the point (x, z).
        """
        # Determine the number of segments
        if N is None:
            N = max(1, round(2000 * self.f * self.b / self.c1))

        # Compute centroid locations for the segments
        xc = np.array([self.b * (-1 + 2 * (j - 0.5) / N) for j in range(1, N + 1)])

        # Initialize pressure
        pressure = 0

        # Initialize Points2DInterface
        points_solver = Points2DInterface(self.c1 / self.c2, self.Dt0, self.b)

        # Sum over all segments
        for segment_center in xc:
            xi = points_solver.calculate_points([x - (e + segment_center) * np.cos(np.radians(self.angt))])[0]

            # Compute distances and angles
            Dtn = self.Dt0 + (e + segment_center) * np.sin(np.radians(self.angt))
            Dxn = x - (e + segment_center) * np.cos(np.radians(self.angt))
            r1 = np.sqrt(xi**2 + Dtn**2) / self.b
            r2 = np.sqrt((Dxn - xi)**2 + z**2) / self.b
            ang1 = np.arcsin(xi / (self.b * r1))
            ang2 = np.arcsin((Dxn - xi) / (self.b * r2))
            ang = np.radians(self.angt) - ang1

            # Avoid division by zero
            ang = ang + np.finfo(float).eps * (ang == 0)

            # Segment directivity
            dir_factor = np.sinc(self.k1b * np.sin(ang) / (np.pi * N))

            # Plane wave transmission coefficient
            Tp = (2 * self.d2 * self.c2 * np.cos(ang1)) / (
                self.d1 * self.c1 * np.cos(ang2) + self.d2 * self.c2 * np.cos(ang1)
            )

            # Phase term and denominator
            phase = np.exp(1j * self.k1b * r1 + 1j * self.k2b * r2)
            denominator = r1 + (self.c2 / self.c1) * r2 * (np.cos(ang1)**2) / (np.cos(ang2)**2)

            # Sum pressure contribution
            pressure += Tp * dir_factor * phase / np.sqrt(denominator)

        # Include external factor
        return pressure * np.sqrt(2 * self.k1b / (1j * np.pi)) / N
