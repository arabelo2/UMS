import numpy as np
from domain.gauss_c15 import GaussC15


class Gauss2D:
    """
    Computes the normalized pressure at a point (x, z) in a fluid for a 1-D element
    using a paraxial multi-Gaussian beam model with 15 Gaussian coefficients.
    """

    def __init__(self, b, f, c):
        """
        Initialize the Gauss2D object.

        Parameters:
            b (float): Half-length of the element (in mm).
            f (float): Frequency (in MHz).
            c (float): Wave speed in the fluid (in m/sec).
        """
        self.b = b
        self.f = f
        self.c = c
        self.kb = 2000 * np.pi * f * b / c  # Wave number
        self.coefficients_a, self.coefficients_b = GaussC15.get_coefficients()

    def compute_pressure(self, x, z):
        """
        Compute the normalized pressure at a point (x, z).

        Parameters:
            x (float or numpy.ndarray): x-coordinate(s) (in mm).
            z (float or numpy.ndarray): z-coordinate(s) (in mm).

        Returns:
            numpy.ndarray: Normalized pressure at the given point(s).
        """
        # Normalize coordinates
        xb = x / self.b
        zb = z / self.b

        # Initialize pressure
        p = np.zeros_like(xb, dtype=np.complex128)

        # Compute pressure using 15 Gaussian beams
        for a, b in zip(self.coefficients_a, self.coefficients_b):
            qb = zb - 1j * 1000 * np.pi * self.f * self.b / (b * self.c)
            qb0 = -1j * 1000 * np.pi * self.f * self.b / (b * self.c)
            p += np.sqrt(qb0 / qb) * a * np.exp(1j * self.kb * xb**2 / (2 * qb))

        return p
