from domain.delay_laws2D_int import DelayLaws2DInterface

class DelayLaws2DInterfaceService:
    """
    Service to compute delay laws for 2D interfaces.
    """

    def compute_delays(self, M, s, angt, ang20, DT0, DF, c1, c2, plt_option='n'):
        """
        Compute time delays for an array in a 2D medium.

        Parameters:
            M (int): Number of elements.
            s (float): Pitch (mm).
            angt (float): Angle of the array with the interface (degrees).
            ang20 (float): Refracted angle in the second medium (degrees).
            DT0 (float): Height of the center of the array above interface (mm).
            DF (float): Depth in the second medium (mm).
            c1 (float): Wave speed in first medium (m/s).
            c2 (float): Wave speed in second medium (m/s).
            plt_option (str, optional): 'y' for plotting rays, 'n' for no plot (default: 'n').

        Returns:
            np.ndarray: Time delays for each array element.
        """
        return DelayLaws2DInterface.compute_delays(M, s, angt, ang20, DT0, DF, c1, c2, plt_option)
