from domain.ls_2Dv import LS2Dv


class LS2DvService:
    """
    Service to calculate normalized pressure for a 2D source.
    """

    def calculate(self, b, f, c, e, x, z, N=None):
        """
        Compute the normalized pressure for the given parameters.

        Parameters:
            b (float): Half-length of the source (in mm).
            f (float): Frequency (in MHz).
            c (float): Wave speed in the fluid (in m/s).
            e (float): Offset of the center of the element along the x-axis (in mm).
            x (numpy.ndarray): x-coordinate(s) (in mm).
            z (numpy.ndarray): z-coordinate(s) (in mm).
            N (int, optional): Number of segments. Defaults to 1 segment per wavelength.

        Returns:
            numpy.ndarray: Normalized pressure at the specified coordinates.
        """
        return LS2Dv.calculate(b, f, c, e, x, z, N)
