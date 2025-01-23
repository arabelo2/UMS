from domain.delay_laws2D import DelayLaws2D


class DelayLaws2DService:
    """
    Service for managing DelayLaws2D computations.
    """

    def compute_delays(self, M, s, Phi, F, c):
        """
        Compute time delays for the array transducer.

        Parameters:
            M (int): Number of elements in the array.
            s (float): Pitch (distance between elements) in mm.
            Phi (float): Beam steering angle in degrees.
            F (float): Focal length in mm (use np.inf for steering only).
            c (float): Wave speed in m/s.

        Returns:
            numpy.ndarray: Array of time delays (in microseconds).
        """
        return DelayLaws2D.compute_delays(M, s, Phi, F, c)
