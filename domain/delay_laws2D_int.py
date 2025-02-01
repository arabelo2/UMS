import numpy as np
import matplotlib.pyplot as plt
from application.ferrari_service import FerrariService  # Ensure correct import

class DelayLaws2DInterface:
    """
    Computes the delay laws for a 1D array in a 2D medium with a planar interface.
    """

    @staticmethod
    def compute_delays(M, s, angt, ang20, DT0, DF, c1, c2, plt_option="n"):
        """
        Compute delay laws for steering and focusing through a planar interface.

        Parameters:
            M (int): Number of elements.
            s (float): Pitch (mm).
            angt (float): Angle of the array with the interface (degrees).
            ang20 (float): Refracted angle in the second medium (degrees).
            DT0 (float): Height of the center of the array above interface (mm).
            DF (float or np.inf): Depth in the second medium (mm).
            c1 (float): Wave speed in first medium (m/s).
            c2 (float): Wave speed in second medium (m/s).
            plt_option (str, optional): 'y' for plotting rays, 'n' for no plot (default: 'n').

        Returns:
            np.ndarray: Time delays for each array element.
        """

        cr = c1 / c2  # Wave speed ratio
        Mb = (M - 1) / 2

        # Compute element centroid positions
        m = np.arange(1, M + 1)
        e = (m - 1 - Mb) * s

        # Compute angles and distances
        ang10 = np.arcsin((c1 / c2) * np.sin(np.radians(ang20)))
        DX0 = DF * np.tan(np.radians(ang20)) + DT0 * np.tan(np.radians(ang10))
        DT = DT0 + e * np.sin(np.radians(angt))
        DX = DX0 - e * np.cos(np.radians(angt))

        # Steering Only Case
        if np.isinf(DF):
            td = np.zeros(M)
            if (np.degrees(ang10) - angt) > 0:
                td = 1000 * (m - 1) * s * np.sin(np.radians(ang10 - angt)) / c1
            else:
                td = 1000 * (M - m) * s * np.abs(np.sin(np.radians(ang10 - angt))) / c1

            # ✅ Only plot if plt_option is explicitly set to 'y'
            if plt_option.lower() == "y":
                DelayLaws2DInterface._plot_rays_steering(M, e, DT, ang10, angt, ang20, DT0)

            return td

        # Steering and Focusing Case
        xi = np.zeros(M)
        r1 = np.zeros(M)
        r2 = np.zeros(M)

        for mm in range(M):
            ferrari_service = FerrariService(cr, DF, DT[mm], DX[mm])
            xi[mm] = ferrari_service.calculate_intersection()
            r1[mm] = np.sqrt(xi[mm]**2 + (DT0 + e[mm] * np.sin(np.radians(angt)))**2)
            r2[mm] = np.sqrt((xi[mm] + e[mm] * np.cos(np.radians(angt)) - DX0)**2 + DF**2)

        t = 1000 * r1 / c1 + 1000 * r2 / c2
        td = np.max(t) - t  # Convert to time delays

        # ✅ Only plot if plt_option is explicitly set to 'y'
        if plt_option.lower() == "y":
            DelayLaws2DInterface._plot_rays_focusing(M, e, DT, xi, DX0, DF, angt)

        return td

    @staticmethod
    def _plot_rays_steering(M, e, DT, ang10, angt, ang20, DT0):
        """
        Plots ray paths for steering only case.
        """
        xp2 = np.zeros((3, M))
        yp2 = np.zeros((3, M))

        for nn in range(M):
            xp2[0, nn] = e[nn] * np.cos(np.radians(angt))
            yp2[0, nn] = DT[nn]
            xp2[1, nn] = e[nn] * np.cos(np.radians(ang10 - angt)) / np.cos(np.radians(ang10)) + DT0 * np.tan(np.radians(ang10))
            yp2[1, nn] = 0
            xp2[2, nn] = xp2[1, nn] + (xp2[1, nn] - xp2[0, nn]) * np.sin(np.radians(ang20))
            yp2[2, nn] = -(xp2[1, nn] - xp2[0, nn]) * np.sin(np.radians(ang20))

        plt.figure()
        plt.plot(xp2, yp2, 'b')
        plt.xlabel("x (mm)")
        plt.ylabel("z (mm)")
        plt.title("Steering Rays")
        plt.show()

    @staticmethod
    def _plot_rays_focusing(M, e, DT, xi, DX0, DF, angt):
        """
        Plots ray paths for steering and focusing case.
        """
        xp = np.zeros((3, M))
        yp = np.zeros((3, M))

        for nn in range(M):
            xp[0, nn] = e[nn] * np.cos(np.radians(angt))
            yp[0, nn] = DT[nn]
            xp[1, nn] = e[nn] * np.cos(np.radians(angt)) + xi[nn]
            yp[1, nn] = 0
            xp[2, nn] = DX0
            yp[2, nn] = -DF

        plt.figure()
        plt.plot(xp, yp, 'b')
        plt.xlabel("x (mm)")
        plt.ylabel("z (mm)")
        plt.title("Steering and Focusing Rays")
        plt.show()
