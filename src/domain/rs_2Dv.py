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
        Compute the normalized pressure at location(s) (x, z) in the fluid.
        The inputs x and z can be scalars or NumPy arrays (for 1D or 2D evaluations).
        This method uses vectorized operations and sums over N segments to approximate
        the Rayleigh–Sommerfeld integral.

        :param x: x-coordinate(s) (mm) where pressure is evaluated.
        :param z: z-coordinate(s) (mm) where pressure is evaluated.
                  For 2D, x and z should be arrays of identical shape (e.g., from meshgrid).
        :param N: (Optional) Number of segments for numerical integration.
                  If not provided, N is computed as round(20000*f*b/c) with a minimum of 1.
        :return: The normalized pressure (a complex number or a NumPy array of complex numbers).
        """
        # Determine the number of segments if not provided.
        if N is None:
            N = round(20000 * self.f * self.b / self.c)
            if N <= 1:
                N = 1

        # Convert inputs to arrays and normalize relative to b.
        xb = np.asarray(x) / self.b
        zb = np.asarray(z) / self.b
        eb = self.e / self.b

        # Add a new axis to xb and zb so that they can broadcast with the integration dimension.
        # After this, if xb originally has shape (m, n) (for 2D) or (m,) (for 1D), it becomes (1, m, n) or (1, m).
        xb = xb[np.newaxis, ...]
        zb = zb[np.newaxis, ...]

        # Compute normalized centroid locations for the segments.
        # For j = 1, 2, ..., N: xc[j] = -1 + 2*(j - 0.5)/N.
        j = np.arange(1, N + 1)
        xc = -1 + 2 * (j - 0.5) / N  # shape (N,)
        # Reshape xc to (N, 1, 1) so that it broadcasts with xb and zb.
        xc = xc.reshape((N, 1, 1))

        # Compute the distance for each segment.
        # The subtraction broadcasts over the integration dimension.
        rb = np.sqrt((xb - xc - eb)**2 + zb**2)  # shape becomes (N, ...) where ... is the shape of xb and zb

        # Evaluate the Hankel function for each segment.
        hankel_values = hankel1(0, self.kb * rb)

        # Sum the contributions over the integration segments (axis 0) and multiply by the external factor.
        p = np.sum(hankel_values, axis=0) * (self.kb / N)
        return p