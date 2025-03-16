from domain.fresnel_int import FresnelIntegral


class FresnelIntegralService:
    """
    Service for managing Fresnel integral computations.
    """

    def compute_integrals(self, x):
        """
        Compute the Fresnel integral for the given input.

        Parameters:
            x (numpy.ndarray): Input array of values.

        Returns:
            numpy.ndarray: Complex Fresnel integral values.
        """
        return FresnelIntegral.compute(x)
