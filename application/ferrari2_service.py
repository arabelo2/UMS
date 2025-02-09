# application/ferrari2_service.py

from domain.ferrari2 import Ferrari2

class Ferrari2Service:
    """
    Application service that wraps the Ferrari2 domain class.
    """
    def __init__(self, cr: float, DF: float, DT: float, DX: float):
        self._solver = Ferrari2(cr, DF, DT, DX)

    def solve(self) -> float:
        """
        Solve for the intersection point xi.
        
        Returns:
            float: The intersection point.
        """
        return self._solver.solve()
