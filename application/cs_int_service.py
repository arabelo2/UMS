from domain.cs_int import CosineSineIntegral


class CosineSineIntegralService:
    """
    Service for managing cosine and sine integral computations.
    """

    def compute_integrals(self, xi):
        """
        Compute the cosine and sine integrals for the given input.

        Parameters:
            xi (numpy.ndarray): Input array of positive values.

        Returns:
            tuple: Cosine integral (c) and sine integral (s).
        """
        return CosineSineIntegral.compute(xi)
