# domain/rs_2Dv.py

import math
import numpy as np
from scipy.special import hankel1

class RayleighSommerfeld2Dv:
    def __init__(self, b: float, f: float, c: float, e: float):
        """
        Initialize the 2-D Rayleigh–Sommerfeld simulation for a 1-D piston source.

        :param b: Half-length of the element (mm); total element length is 2*b.
        :param f: Frequency (MHz).
        :param c: Wave speed in the fluid (m/s).
        :param e: Lateral offset of the element's center along the x-axis (mm).
        """
        self.b = b
        self.f = f
        self.c = c
        self.e = e
        # Compute wave number factor: kb = 2000 * pi * b * f / c
        self.kb = 2000 * math.pi * self.b * self.f / self.c

    def compute_pressure(self, x, z, N: int = None):
        """
        Compute the normalized pressure at a location (x, z) in the fluid.
        The inputs x and z can be scalars or NumPy arrays.
        
        :param x: x-coordinate (mm) where pressure is evaluated.
        :param z: z-coordinate (mm) where pressure is evaluated (can be an array).
        :param N: (Optional) Number of segments for numerical integration.
                  If not provided, N is computed as round(20000 * f * b / c) with a minimum of 1.
        :return: The normalized pressure (complex). If z is an array, returns a NumPy array.
        """
        # Determine the number of segments if not provided.
        if N is None:
            N = round(20000 * self.f * self.b / self.c)
            if N <= 1:
                N = 1

        # Ensure x and z are NumPy arrays.
        x = np.atleast_1d(x)
        z = np.atleast_1d(z)

        # Normalize positions relative to b.
        xb = x / self.b      # scalar or 1-D array
        zb = z / self.b      # 1-D array (for example, 500 values)
        eb = self.e / self.b # scalar

        # Compute normalized centroid locations for the segments.
        # For j = 1, 2, ..., N: xc[j] = -1 + 2*(j - 0.5)/N.
        j = np.arange(1, N + 1)
        xc = -1 + 2 * (j - 0.5) / N  # shape (N,)
        # Reshape xc to (N, 1) so that it broadcasts with zb.
        xc = xc.reshape((N, 1))

        # Compute the distance for each segment.
        # Here, (xb - xc - eb) has shape (N, len(xb)) (but if x is scalar, shape (N,1)),
        # and zb is shape (len(z),), so we force broadcasting by ensuring zb is 1-D.
        # (If x is scalar, then xb and eb are scalars.)
        rb = np.sqrt((xb - xc - eb)**2 + zb**2)  # shape (N, len(z)) if x is scalar

        # Evaluate the Hankel function (MATLAB's besselh(0,1,...) is scipy.special.hankel1(0, ...)).
        hankel_values = hankel1(0, self.kb * rb)

        # Sum over the segments (axis 0) and apply the external factor.
        p = np.sum(hankel_values, axis=0) * (self.kb / N)
        return p
