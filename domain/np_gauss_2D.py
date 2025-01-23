import numpy as np
from domain.gauss_c10 import GaussC10


class NPGauss2D:
    @staticmethod
    def calculate(b, f, c, e, x, z):
        """
        Calculate the normalized pressure field.

        Parameters:
            b (float): Half-length of the element (in mm).
            f (float): Frequency (in MHz).
            c (float): Wave speed in the fluid (in m/s).
            e (float): Offset of the center of the element in the x-direction (in mm).
            x (float or numpy.ndarray): x-coordinate(s) (in mm).
            z (float or numpy.ndarray): z-coordinate(s) (in mm).

        Returns:
            numpy.ndarray: Normalized pressure field.
        """
        # Get Gaussian coefficients
        A, B = GaussC10.get_coefficients()

        # Define non-dimensional quantities
        xb = x / b
        zb = z / b
        eb = e / b
        Rb = np.sqrt((xb - eb)**2 + zb**2)
        kb = 2000 * np.pi * f * b / c
        Db = kb / 2
        cosp = zb / Rb

        # Initialize normalized pressure
        p = np.zeros_like(zb, dtype=np.complex128)

        # Calculate normalized pressure field
        for nn in range(10):
            arg = cosp**2 + 1j * B[nn] * Rb / Db
            Dn = np.sqrt(arg)
            amp = A[nn] * np.exp(1j * kb * Rb) / Dn
            p += amp * np.exp(-1j * kb * xb**2 / (2 * Rb * arg))

        return p
