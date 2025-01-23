from domain.ferrari2 import FerrariSolver


class Points2DInterface:
    """
    Computes intersection points of a 2D interface using Ferrari's method.
    """

    def __init__(self, wave_speed_ratio, depth, height):
        """
        Initialize the 2D interface solver with the required parameters.

        Parameters:
            wave_speed_ratio (float): Ratio of wave speeds (c1/c2).
            depth (float): Depth of the point in medium two.
            height (float): Height of the point in medium one.
        """
        self.wave_speed_ratio = wave_speed_ratio
        self.depth = depth
        self.height = height

    def calculate_points(self, distances):
        """
        Compute the intersection points for given separation distances.

        Parameters:
            distances (list or numpy.ndarray): Separation distances (DX values).

        Returns:
            list: Calculated intersection points.
        """
        points = []
        for DX in distances:
            solver = FerrariSolver(self.wave_speed_ratio, self.depth, self.height, DX)
            xi = solver.solve()
            points.append(xi)
        return points
