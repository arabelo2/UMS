import numpy as np


class CosineSineIntegral:
    """
    Computes approximations of the cosine and sine integrals for positive values of input
    based on Abramowitz and Stegun's formulas.
    """

    @staticmethod
    def compute(xi):
        """
        Compute cosine and sine integrals for positive values of xi.

        Parameters:
            xi (numpy.ndarray): Input array of positive values.

        Returns:
            tuple: Cosine integral (c) and sine integral (s) as numpy arrays.
        """
        if np.any(xi < 0):
            raise ValueError("Input values for cs_int must be non-negative.")

        # Compute f and g functions
        f = (1 + 0.926 * xi) / (2 + 1.792 * xi + 3.104 * xi**2)
        g = 1 / (2 + 4.142 * xi + 3.492 * xi**2 + 6.67 * xi**3)

        # Compute cosine and sine integrals
        c = 0.5 + f * np.sin(np.pi * xi**2 / 2) - g * np.cos(np.pi * xi**2 / 2)
        s = 0.5 - f * np.cos(np.pi * xi**2 / 2) - g * np.sin(np.pi * xi**2 / 2)

        return c, s
