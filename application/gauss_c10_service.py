from domain.gauss_c10 import GaussC10


class GaussC10Service:
    """
    Service to provide access to the Wen and Breazeale coefficients.
    """

    def get_coefficients(self):
        """
        Returns the coefficients a and b.

        Returns:
            tuple: (a, b) where:
                - a (numpy.ndarray): Array of complex coefficients 'a'.
                - b (numpy.ndarray): Array of complex coefficients 'b'.
        """
        return GaussC10.get_coefficients()
