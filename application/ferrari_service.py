from domain.ferrari2 import FerrariSolver


class FerrariService:
    """
    Service layer to manage Ferrari solver interactions.
    """

    def __init__(self, cr, DF, DT, DX):
        self.solver = FerrariSolver(cr, DF, DT, DX)

    def calculate_intersection(self):
        """
        Calculate the intersection point using the Ferrari solver.

        Returns:
            float: The calculated intersection point.
        """
        return self.solver.solve()
