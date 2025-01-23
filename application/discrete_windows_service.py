from domain.discrete_windows import DiscreteWindows


class DiscreteWindowsService:
    """
    Service to manage computations of discrete apodization amplitudes.
    """

    def get_amplitudes(self, M, window_type):
        """
        Get the apodization amplitudes for a given window type.

        Parameters:
            M (int): Number of elements.
            window_type (str): Type of window ('cos', 'Han', 'Ham', 'Blk', 'tri', 'rect').

        Returns:
            numpy.ndarray: Array of apodization amplitudes.
        """
        return DiscreteWindows.compute(M, window_type)
