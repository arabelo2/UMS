from domain.ls_2Dint import LS2DInterface


class LS2DInterfaceService:
    """
    Service for managing LS2DInterface computations.
    """

    def __init__(self, b, f, mat, angt, Dt0):
        self.solver = LS2DInterface(b, f, mat, angt, Dt0)

    def calculate_pressure(self, x, z, e, N=None):
        """
        Compute the normalized pressure for given parameters.

        Parameters:
            x (float): x-coordinate.
            z (float): z-coordinate.
            e (float): Offset.
            N (int, optional): Number of segments.

        Returns:
            complex: Computed pressure.
        """
        return self.solver.compute_pressure(x, z, e, N)
