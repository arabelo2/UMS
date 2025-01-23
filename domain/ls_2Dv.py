import numpy as np


class LS2Dv:
    @staticmethod
    def calculate(b, f, c, e, x, z, N=None):
        """
        Calculate the normalized pressure.

        Parameters:
            b (float): Half-length of the source (in mm).
            f (float): Frequency (in MHz).
            c (float): Wave speed in the fluid (in m/s).
            e (float): Offset of the center of the element along the x-axis (in mm).
            x (float or numpy.ndarray): x-coordinate(s) (in mm).
            z (numpy.ndarray): z-coordinate(s) (in mm).
            N (int, optional): Number of segments. Defaults to 1 segment per wavelength.

        Returns:
            numpy.ndarray: Normalized pressure at the specified coordinates.
        """
        # Compute wave number
        kb = 2000 * np.pi * b * f / c

        # Determine number of segments if not specified
        if N is None:
            N = max(1, round(2000 * f * b / c))

        # Normalize positions
        xb = x / b
        zb = z / b
        eb = e / b

        # Compute normalized centroid locations for the segments
        xc = np.array([-1 + 2 * (jj - 0.5) / N for jj in range(1, N + 1)])

        # Initialize pressure as a zero array with the same shape as z
        p = np.zeros_like(z, dtype=np.complex128)

        # Sum over all segments
        for xc_k in xc:
            ang = np.arctan((xb - xc_k - eb) / zb)
            ang += np.finfo(float).eps * (ang == 0)  # Prevent division by zero
            dir_factor = np.sinc(kb * np.sin(ang) / np.pi / N)
            rb = np.sqrt((xb - xc_k - eb)**2 + zb**2)
            phase = np.exp(1j * kb * rb)
            p += dir_factor * phase / np.sqrt(rb)

        # Include external factor
        return p * (np.sqrt(2 * kb / (1j * np.pi))) / N
