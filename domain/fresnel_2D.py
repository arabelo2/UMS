import numpy as np
from domain.fresnel_int import FresnelIntegral


class Fresnel2D:
    """
    Computes the normalized pressure field at a point (x, z) for a 1-D element
    radiating into a fluid using Fresnel integrals.
    """

    def __init__(self, b, f, c):
        """
        Initialize the Fresnel2D object.

        Parameters:
            b (float): Half-length of the element (in mm).
            f (float): Frequency (in MHz).
            c (float): Wave speed in the fluid (in m/sec).
        """
        self.b = b
        self.f = f
        self.c = c
        self.kb = 2000 * np.pi * f * b / c  # Wave number

    def compute_pressure(self, x, z):
        """
        Compute the normalized pressure field at a given point (x, z).

        Parameters:
            x (float or numpy.ndarray): x-coordinate(s) (in mm).
            z (float or numpy.ndarray): z-coordinate(s) (in mm).

        Returns:
            numpy.ndarray: Normalized pressure at the given point(s).
        """
        # Normalize coordinates
        xb = x / self.b
        zb = z / self.b

        # Argument for the Fresnel integral
        arg = np.sqrt(self.kb / (np.pi * zb))

        # Handle 2D grids
        arg_xb1 = (arg * (xb + 1)).flatten()
        arg_xb2 = (arg * (xb - 1)).flatten()

        # Compute Fresnel integrals and reshape to original grid shape
        fresnel_xb1 = FresnelIntegral.compute(arg_xb1).reshape(x.shape)
        fresnel_xb2 = FresnelIntegral.compute(arg_xb2).reshape(x.shape)

        # Compute normalized pressure
        fresnel_term = fresnel_xb1 - fresnel_xb2
        pressure = np.sqrt(1 / (2j)) * np.exp(1j * self.kb * zb) * fresnel_term

        return pressure

