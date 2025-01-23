from domain.pts_2Dintf import Points2DInterface


class Points2DInterfaceService:
    """
    Service to manage the calculation of 2D interface points.
    """

    def __init__(self, wave_speed_ratio, depth, height):
        self.interface = Points2DInterface(wave_speed_ratio, depth, height)

    def compute(self, distances):
        """
        Compute 2D interface points.

        Parameters:
            distances (list or numpy.ndarray): Separation distances (DX values).

        Returns:
            list: Computed 2D interface points.
        """
        return self.interface.calculate_points(distances)
