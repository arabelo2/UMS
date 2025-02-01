from domain.delay_laws2D_int import DelayLaws2DInterface

class DelayLaws2DInterfaceService:
    """
    Service to compute delay laws for 2D interfaces.
    """

    def compute_delays(self, M, s, angt, ang20, DT0, DF, c1, c2, plt_option='n'):
        """
        Compute time delays for an array in a 2D medium.
        """
        return DelayLaws2DInterface.compute_delays(M, s, angt, ang20, DT0, DF, c1, c2, plt_option)
