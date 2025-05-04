# domain/ls_2Dv.py

import math
import numpy as np

class LS2Dv:
    def __init__(self, b: float, f: float, c: float, e: float):
        """
        Initialize the LS2Dv simulation for a two-dimensional source.
        
        :param b: Half-length of the element (mm). The element length is 2*b.
        :param f: Frequency (MHz).
        :param c: Wave speed (m/s).
        :param e: Lateral offset of the element's center (mm).
        """
        self.b = b
        self.f = f
        self.c = c
        self.e = e
        # Compute wave number: kb = 2000 * pi * b * f / c
        self.kb = 2000 * math.pi * self.b * self.f / self.c

    def compute_pressure(self, x, z, N: int = None):
        """
        Compute the normalized pressure at location(s) (x, z) in the fluid.
        This function uses an asymptotic approximation for the Hankel function for large wave numbers.
        
        If N (the number of segments) is not provided, it is computed automatically using:
            N = round(2000 * f * b / c)
        which corresponds to using one segment per wavelength.
        
        :param x: x-coordinate(s) (mm) where pressure is evaluated.
        :param z: z-coordinate(s) (mm) where pressure is evaluated.
                  Can be scalar or an array (e.g. from meshgrid for 2D simulation).
        :param N: Optional number of segments for numerical integration.
                  If provided, that value is used; otherwise, it is computed automatically.
        :return: Normalized pressure (complex number or NumPy array).
        """
        if N is None:
            N = round(2000 * self.f * self.b / self.c)
            if N < 1:
                N = 1
        
        # Normalize coordinates
        xb = np.atleast_1d(x) / self.b
        zb = np.atleast_1d(z) / self.b
        eb = self.e / self.b
        
        # Add a new axis so that the integration (over segments) broadcasts properly.
        xb = xb[np.newaxis, ...]
        zb = zb[np.newaxis, ...]
        
        # Compute normalized centroid positions for the segments.
        j = np.arange(1, N + 1)
        xc = -1 + 2*(j - 0.5)/N  # shape (N,)
        xc = xc.reshape((N, 1, 1))
        
        p = 0
        eps = np.finfo(float).eps
        for kk in range(N):
            # Calculate angle for the directivity factor
            with np.errstate(divide="ignore", invalid="ignore"):
                ang = np.arctan((xb - xc[kk] - eb) / zb)
            # Add a tiny epsilon to avoid division by zero when ang is zero
            ang = ang + eps * (ang == 0)
            denom = self.kb * np.sin(ang) / N
            dir_factor = np.sin(denom) / denom
            rb = np.sqrt((xb - xc[kk] - eb)**2 + zb**2)
            p = p + dir_factor * np.exp(1j * self.kb * rb) / np.sqrt(rb)
        # Include external factor.
        p = p * (np.sqrt(2 * self.kb / (1j * np.pi))) / N
        return p
