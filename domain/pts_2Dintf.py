# domain/pts_2Dintf.py

import numpy as np
import math
from application.ferrari2_service import Ferrari2Service
from application.init_xi_service import InitXiService

class Pts2DIntf:
    """
    Domain class for computing the intersection points (xi) along a plane interface.
    
    Parameters:
      e    : offset of the element from the array center (mm)
      xn   : offset of the segment from the element center (mm)
      angt : angle (in degrees) that the array makes with the x-axis
      Dt0  : distance of the array center above the interface (mm)
      c1   : wave speed in medium 1 (m/s)
      c2   : wave speed in medium 2 (m/s)
      x, z : coordinates (mm) of the target point in medium 2
    """
    def __init__(self, e: float, xn: float, angt: float, Dt0: float,
                 c1: float, c2: float, x, z):
        self.e = e
        self.xn = xn
        self.angt = angt
        self.Dt0 = Dt0
        self.c1 = c1
        self.c2 = c2
        self.x = x
        self.z = z

    def compute(self):
        """
        Compute the intersection points xi.
        
        Returns:
          xi : a NumPy array (with shape determined by broadcasting x and z)
               containing the intersection points (in mm).
               
        Note:
          The output is squeezed to remove any singleton dimensions so that if one of the inputs
          is a vector, xi is returned as a 1-D array.
        """
        # Calculate wave speed ratio.
        cr = self.c1 / self.c2

        # Get the broadcast shape and initialize xi using the InitXiService.
        xi, P, Q = InitXiService(self.x, self.z).compute()

        # Force x and z to have the same shape as xi using np.broadcast_to.
        x_b = np.broadcast_to(np.array(self.x), (P, Q))
        z_b = np.broadcast_to(np.array(self.z), (P, Q))
        
        # Compute the effective vertical distance (common for all points).
        Dtn = self.Dt0 + (self.e + self.xn) * math.sin(math.radians(self.angt))
        
        # Iterate over all indices in the (P, Q) shape.
        for index, x_val in np.ndenumerate(x_b):
            # For each point, use the corresponding z value.
            current_z = z_b[index]
            # Compute the horizontal distance.
            Dxn = x_val - (self.e + self.xn) * math.cos(math.radians(self.angt))
            # Create a Ferrari2Service instance.
            # Mapping: DF = current_z, DT = Dtn, DX = Dxn.
            ferrari_service = Ferrari2Service(cr, current_z, Dtn, Dxn)
            xi_val = ferrari_service.solve()
            xi[index] = xi_val
        
        # Squeeze the result to remove singleton dimensions (e.g., (1, 3) becomes (3,))
        return np.squeeze(xi)
