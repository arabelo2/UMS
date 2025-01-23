from domain.on_axis_foc2D import OnAxisFoc2D


class OnAxisFoc2DService:
    """
    Service for managing OnAxisFoc2D computations.
    """

    def __init__(self, b, R, f, c):
        self.solver = OnAxisFoc2D(b, R, f, c)

    def compute_pressure(self, z):
        """
        Compute the on-axis normalized pressure for the given distances.

        Parameters:
            z (float or numpy.ndarray): On-axis distances (in mm).

        Returns:
            numpy.ndarray: Normalized on-axis pressure at the given distances.
        """
        return self.solver.compute_pressure(z)
