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
        ratio = cr * np.sin(np.radians(ang20))
        if ratio > 1:
            raise ValueError(
                f"Invalid input: (c1/c2) * sin(ang20) = {ratio:.3f} > 1. "
                "Please adjust the input parameters."
            )

        Mb = (M - 1) / 2.0
        m = np.arange(1, M + 1)
        e = (m - 1 - Mb) * s
        DT_arr = DT0 + e * np.sin(np.radians(angt))
        ang10 = np.degrees(np.arcsin((c1 / c2) * np.sin(np.radians(ang20))))
        
        if np.isinf(DF):
            if (ang10 - angt) > 0:
                td = 1000 * (m - 1) * s * np.sin(np.radians(ang10 - angt)) / c1
            else:
                td = 1000 * (M - m) * s * abs(np.sin(np.radians(ang10 - angt))) / c1
            td = np.array(td)
            return td
        else:
            DX0 = DF * np.tan(np.radians(ang20)) + DT0 * np.tan(np.radians(ang10))
            DX = DX0 - e * np.cos(np.radians(angt))
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

    def compute_delays_and_rays(self):
        """
        Compute the delays and the ray geometry for plotting.
        Returns:
            td (numpy.ndarray): The computed delays.
            (xp, yp) (tuple): Two arrays (each shape (3, M)) containing the x and y
                              coordinates for three key points per element:
                              [element center, interface intersection, focal point].
        """
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
        ratio = cr * np.sin(np.radians(ang20))
        if ratio > 1:
            raise ValueError(
                f"Invalid input: (c1/c2) * sin(ang20) = {ratio:.3f} > 1. "
                "Please adjust the input parameters."
            )
        
        Mb = (M - 1) / 2.0
        m = np.arange(1, M + 1)
        e = (m - 1 - Mb) * s
        DT_arr = DT0 + e * np.sin(np.radians(angt))
        ang10 = np.degrees(np.arcsin((c1 / c2) * np.sin(np.radians(ang20))))
        
        # Prepare arrays for ray geometry (3 key points per element)
        xp = np.zeros((3, M))
        yp = np.zeros((3, M))
        
        if np.isinf(DF):
            # Steering-only case
            if (ang10 - angt) > 0:
                td = 1000 * (m - 1) * s * np.sin(np.radians(ang10 - angt)) / c1
            else:
                td = 1000 * (M - m) * s * abs(np.sin(np.radians(ang10 - angt))) / c1
            td = np.array(td)
            # Compute ray geometry using the MATLAB steering-only logic
            for nn in range(M):
                # Starting point: element centroid in medium 1
                xp[0, nn] = e[nn] * np.cos(np.radians(angt))
                yp[0, nn] = DT_arr[nn]
                # Intersection with the interface
                xp[1, nn] = e[nn] * np.cos(np.radians(ang10 - angt)) / np.cos(np.radians(ang10)) + DT0 * np.tan(np.radians(ang10))
                yp[1, nn] = 0
                dm = e[nn] * np.cos(np.radians(ang10 - angt)) / np.cos(np.radians(ang10))
                if ang20 > 0:
                    dM = e[-1] * np.cos(np.radians(ang10 - angt)) / np.cos(np.radians(ang10))
                else:
                    dM = e[0] * np.cos(np.radians(ang10 - angt)) / np.cos(np.radians(ang10))
                xp[2, nn] = xp[1, nn] + (dM - dm) * (np.sin(np.radians(ang20)) ** 2)
                yp[2, nn] = -(dM - dm) * np.sin(np.radians(ang20)) * np.cos(np.radians(ang20))
            return td, (xp, yp)
        else:
            # Steering and focusing case.
            DX0 = DF * np.tan(np.radians(ang20)) + DT0 * np.tan(np.radians(ang10))
            DX = DX0 - e * np.cos(np.radians(angt))
            xi = np.zeros(M)
            r1 = np.zeros(M)
            r2 = np.zeros(M)
            for mm in range(M):
                xi[mm] = ferrari2(cr, DF, DT_arr[mm], DX[mm])
                r1[mm] = np.sqrt(xi[mm]**2 + (DT0 + e[mm] * np.sin(np.radians(angt)))**2)
                r2[mm] = np.sqrt((xi[mm] + e[mm] * np.cos(np.radians(angt)) - DX0)**2 + DF**2)
            t = 1000 * r1 / c1 + 1000 * r2 / c2
            td = np.max(t) - t
            # Compute ray geometry for focusing case:
            for i in range(M):
                # Point 1: Element centroid in medium 1
                x_element = e[i] * np.cos(np.radians(angt))
                y_element = DT_arr[i]
                # Point 2: Intersection at the interface
                x_interface = x_element + xi[i]
                y_interface = 0
                # Point 3: Focal point in medium 2
                x_focus = DX0
                y_focus = -DF
                xp[0, i] = x_element
                yp[0, i] = y_element
                xp[1, i] = x_interface
                yp[1, i] = y_interface
                xp[2, i] = x_focus
                yp[2, i] = y_focus
            return td, (xp, yp)
