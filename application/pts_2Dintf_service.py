# application/pts_2Dintf_service.py

from domain.pts_2Dintf import Pts2DIntf

class Pts2DIntfService:
    """
    Application service that wraps the Pts2DIntf domain object.
    """
    def __init__(self, e: float, xn: float, angt: float, Dt0: float,
                 c1: float, c2: float, x, z):
        self._pts_intf = Pts2DIntf(e, xn, angt, Dt0, c1, c2, x, z)

    def compute(self):
        """
        Compute the intersection points xi.
        
        Returns:
          numpy array: The computed xi values (in mm)
        """
        return self._pts_intf.compute()
