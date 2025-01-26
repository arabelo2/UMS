from domain.rs_2Dv import RS2Dv


class RS2DvService:
    """
    Service to calculate normalized pressure using RS2Dv.
    """

    def calculate(self, b, f, c, e, x, z, Nopt=None):
        """
        Compute the normalized pressure for the given parameters.

        Parameters:
            b (float): Half-length of the element (in mm).
            f (float): Frequency (in MHz).
            c (float): Wave speed in the fluid (in m/s).
            e (float): Offset of the center of the element along the x-axis (in mm).
            x (float or numpy.ndarray): x-coordinate(s) (in mm).
            z (float or numpy.ndarray): z-coordinate(s) (in mm).
            Nopt (int, optional): Number of segments. Defaults to 10 segments per wavelength.

        Returns:
            numpy.ndarray: Normalized pressure at the specified coordinates.
        """
        return RS2Dv.calculate(b, f, c, e, x, z, Nopt)
