# domain/delay_laws2D_int.py

import numpy as np
import math
from domain.ferrari2 import ferrari2

class DelayLaws2DInt:
    """
    Class to compute delay laws for steering and focusing an array of 1-D elements
    through a planar interface between two media in two dimensions.
    """
    def __init__(self, M, s, angt, ang20, DT0, DF, c1, c2, plt_option='n'):
        """
        Initialize delay law parameters.
        
        Parameters:
            M         (int)  : Number of elements.
            s         (float): Array pitch (mm).
            angt      (float): Array angle with the interface (degrees).
            ang20     (float): Refracted angle in the second medium (degrees).
            DT0       (float): Height of the center of the array above the interface (mm).
            DF        (float): Depth in the second medium (mm); use inf for steering-only.
            c1        (float): Wave speed in the first medium (m/s).
            c2        (float): Wave speed in the second medium (m/s).
            plt_option(str)  : 'y' to indicate that plotting is desired, 'n' otherwise.
                              (Note: Plotting should be handled in the interface layer.)
        """
        self.M = M
        self.s = s
        self.angt = angt
        self.ang20 = ang20
        self.DT0 = DT0
        self.DF = DF
        self.c1 = c1
        self.c2 = c2
        self.plt_option = plt_option

    def compute_delays(self):
        
        # Validate input: M must be greater than zero.
        if self.M <= 0:
            raise ValueError("Number of elements M must be greater than zero.")
        M = self.M
        
        s = self.s
        angt = self.angt
        ang20 = self.ang20
        DT0 = self.DT0
        DF = self.DF
        c1 = self.c1
        c2 = self.c2

        cr = c1 / c2
        Mb = (M - 1) / 2.0
        m = np.arange(1, M + 1)  # m = 1, 2, ..., M
        # Compute element centroids
        e = (m - 1 - Mb) * s

        # Compute incident central ray angle in medium 1 (in degrees)
        ang10 = np.degrees(np.arcsin((c1/c2) * np.sin(np.radians(ang20))))
        # Compute DX0: distance along interface from array center to focal point
        # When DF is infinite and ang20 is 0, set DX0 to DT0 * tan(ang10) to avoid inf*0 issues.
        if np.isinf(DF):
            DX0 = DT0 * np.tan(np.radians(ang10))
        else:
            DX0 = DF * np.tan(np.radians(ang20)) + DT0 * np.tan(np.radians(ang10))
        # Heights of elements above the interface in medium 1
        DT_arr = DT0 + e * np.sin(np.radians(angt))
        # Distances along the interface from elements to the focal point
        DX = DX0 - e * np.cos(np.radians(angt))
        
        if np.isinf(DF):
            # Steering-only case: use linear law
            if (ang10 - angt) > 0:
                td = 1000 * (m - 1) * s * np.sin(np.radians(ang10 - angt)) / c1
            else:
                td = 1000 * (M - m) * s * abs(np.sin(np.radians(ang10 - angt))) / c1
            td = np.array(td)
            return td
        
        else:
            # Steering and focusing case.
            xi = np.zeros(M)
            r1 = np.zeros(M)
            r2 = np.zeros(M)
            for mm in range(M):
                xi[mm] = ferrari2(cr, DF, DT_arr[mm], DX[mm])
                r1[mm] = np.sqrt(xi[mm]**2 + (DT0 + e[mm] * np.sin(np.radians(angt)))**2)
                r2[mm] = np.sqrt((xi[mm] + e[mm] * np.cos(np.radians(angt)) - DX0)**2 + DF**2)
            t = 1000 * r1 / c1 + 1000 * r2 / c2
            td = np.max(t) - t
            return td
