from domain.fresnel_2D import Fresnel2D


class Fresnel2DService:
    """
    Service for managing Fresnel2D computations.
    """

    def __init__(self, b, f, c):
        self.solver = Fresnel2D(b, f, c)

    def compute_pressure(self, x, z):
        """
        Compute the normalized pressure field at a given point or grid.

        Parameters:
            x (float or numpy.ndarray): x-coordinate(s) (in mm).
            z (float or numpy.ndarray): z-coordinate(s) (in mm).

        Returns:
            numpy.ndarray: Normalized pressure at the given point(s).
        """
        return self.solver.compute_pressure(x, z)
