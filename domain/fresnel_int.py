import numpy as np
from domain.cs_int import CosineSineIntegral


class FresnelIntegral:
    """
    Computes the Fresnel integral defined as the integral from t = 0 to t = x of
    the function exp(i * pi * t^2 / 2). Uses cosine and sine integrals from Abramowitz and Stegun.
    """

    @staticmethod
    def compute(x):
        """
        Compute the Fresnel integral for an array of values.

        Parameters:
            x (numpy.ndarray): Input array of values.

        Returns:
            numpy.ndarray: Complex Fresnel integral values.
        """
        # Separate arguments into positive and negative values
        xn = -x[x < 0]  # Negative values (made positive)
        xp = x[x >= 0]  # Positive values

        # Compute cosine and sine integrals for negative values
        cn, sn = CosineSineIntegral.compute(xn)
        cn = -cn  # Adjust signs for negative values
        sn = -sn

        # Compute cosine and sine integrals for positive values
        cp, sp = CosineSineIntegral.compute(xp)

        # Combine results
        ct = np.concatenate([cn, cp])  # Cosine integrals
        st = np.concatenate([sn, sp])  # Sine integrals

        # Return complex Fresnel integral
        return ct + 1j * st
