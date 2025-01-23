from domain.gauss_2D import Gauss2D


class Gauss2DService:
    """
    Service for managing Gauss2D computations.
    """

    def __init__(self, b, f, c):
        self.solver = Gauss2D(b, f, c)

    def compute_pressure(self, x, z):
        """
        Compute the normalized pressure at a point or grid.

        Parameters:
            x (float or numpy.ndarray): x-coordinate(s) (in mm).
            z (float or numpy.ndarray): z-coordinate(s) (in mm).

        Returns:
            numpy.ndarray: Normalized pressure at the given point(s).
        """
        return self.solver.compute_pressure(x, z)
