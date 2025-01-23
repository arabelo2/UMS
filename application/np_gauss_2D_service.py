from domain.np_gauss_2D import NPGauss2D


class NPGauss2DService:
    """
    Service to compute the normalized pressure field using NPGauss2D.
    """

    def calculate(self, b, f, c, e, x, z):
        """
        Compute the normalized pressure field.

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
        return NPGauss2D.calculate(b, f, c, e, x, z)
