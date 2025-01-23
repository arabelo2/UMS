from domain.gauss_c15 import GaussC15

class GaussC15Service:
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
        return GaussC15.get_coefficients()
