import numpy as np
from scipy.special import hankel1


class RS2Dv:
    """
    Computes the normalized pressure at a location (x, z) in a fluid for a 1-D element
    using the Rayleigh-Sommerfeld integral.
    """

    @staticmethod
    def calculate(b, f, c, e, x, z, Nopt=None):
        """
        Calculate the normalized pressure.

        Parameters:
            b (float): Half-length of the element (in mm).
            f (float): Frequency (in MHz).
            c (float): Wave speed in the fluid (in m/s).
            e (float): Offset of the center of the element along the x-axis (in mm).
            x (float or numpy.ndarray): x-coordinate(s) (in mm).
            z (float or numpy.ndarray): z-coordinate(s) (in mm).
            Nopt (int, optional): Number of segments. If None, it defaults to 10 segments per wavelength.

        Returns:
            numpy.ndarray: Normalized pressure at the specified coordinates.
        """
        # Compute wave number
        kb = 2000 * np.pi * b * f / c

        # Determine number of segments
        if Nopt is not None:
            N = Nopt
        else:
            N = max(1, round(20000 * f * b / c))  # Default: 10 segments per wavelength

        # Use normalized positions
        xb = x / b
        zb = z / b
        eb = e / b

        # Compute normalized centroid locations for the segments
        xc = np.array([-1 + 2 * (jj - 0.5) / N for jj in range(1, N + 1)])

        # Calculate normalized pressure
        p = np.zeros_like(xb, dtype=np.complex128)
        for xc_k in xc:
            rb = np.sqrt((xb - xc_k - eb)**2 + zb**2)
            p += hankel1(0, kb * rb)

        # Include external factor
        return p * (kb / N)
