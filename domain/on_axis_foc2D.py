import numpy as np
from domain.fresnel_int import FresnelIntegral


class OnAxisFoc2D:
    """
    Computes the on-axis normalized pressure for a 1-D focused piston element.
    """

    def __init__(self, b, R, f, c):
        """
        Initialize the focused piston element parameters.

        Parameters:
            b (float): Transducer half-length (in mm).
            R (float): Focal length (in mm).
            f (float): Frequency (in MHz).
            c (float): Wave speed of the surrounding fluid (in m/sec).
        """
        self.b = b
        self.R = R
        self.f = f
        self.c = c
        self.kb = 2000 * np.pi * f * b / c  # Transducer wave number

    def compute_pressure(self, z):
        """
        Compute the on-axis normalized pressure for the given distances z.

        Parameters:
            z (float or numpy.ndarray): On-axis distances (in mm).

        Returns:
            numpy.ndarray: Normalized on-axis pressure at the given distances.
        """
        z = np.array(z) + np.finfo(float).eps * (z == 0)  # Prevent division by zero

        # Define `u` and prevent invalid values
        u = 1 - z / self.R
        u[u == 0] += np.finfo(float).eps  # Handle cases where u = 0

        # Masks for near-focus and far-focus regions
        mask_near = np.abs(u) <= 0.005
        mask_far = ~mask_near

        # Initialize the pressure array
        pressure = np.zeros_like(z, dtype=np.complex128)

        # Compute for near-focus region
        if np.any(mask_near):
            fresnel_near = np.sqrt(2 / 1j) * np.sqrt((self.b / self.R) * self.kb / np.pi)
            pressure[mask_near] = fresnel_near

        # Compute for far-focus region
        if np.any(mask_far):
            x_far = np.sqrt(np.abs((-u[mask_far] * self.kb * self.b) / (np.pi * z[mask_far])))
            fresnel_far = np.sqrt(2 / 1j) * FresnelIntegral.compute(x_far) / np.sqrt(np.abs(u[mask_far]))
            pressure[mask_far] = fresnel_far

        return pressure

