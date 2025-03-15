# domain/delay_laws3Dint.py

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # ensures 3D plotting works
from domain.ferrari2 import ferrari2

def delay_laws3Dint(Mx: int, My: int, sx: float, sy: float,
                    theta: float, phi: float, theta20: float,
                    DT0: float, DF: float, c1: float, c2: float,
                    plt_option: str = 'n') -> np.ndarray:
    """
    Compute the delay laws (in microseconds) for steering and focusing a 2-D array 
    through a planar interface between two media in three dimensions.

    Parameters:
        Mx, My   : int
                   Number of elements in the (x', y') directions.
        sx, sy   : float
                   Pitches (in mm) along the x' and y' directions.
        theta    : float
                   Angle (in degrees) that the array makes with the interface.
        phi      : float
                   Steering parameter (in degrees) for the second medium.
        theta20  : float
                   Steering (refracted) angle in the second medium (in degrees).
        DT0      : float
                   Height (in mm) of the array center above the interface.
        DF       : float
                   Focal distance (in mm) in the second medium. Use np.inf for steering-only.
        c1, c2   : float
                   Wave speeds (in m/s) for the first and second media, respectively.
        plt_option: str
                   'y' to plot the ray geometry, 'n' otherwise.
                   
    Returns:
        td : np.ndarray
             A 2D array of time delays (in microseconds) with shape (Mx, My).
    """
    # Compute wave speed ratio
    cr = c1 / c2

    # Compute element centroid locations
    Mbx = (Mx - 1) / 2.0
    Mby = (My - 1) / 2.0
    m_indices = np.arange(1, Mx + 1)  # MATLAB: 1:Mx
    ex = (m_indices - 1 - Mbx) * sx   # vector of length Mx
    n_indices = np.arange(1, My + 1)   # 1:My
    ey = (n_indices - 1 - Mby) * sy    # vector of length My

    # Initialize arrays
    t = np.zeros((Mx, My))
    Db = np.zeros((Mx, My))
    De = np.zeros(Mx)
    xi = np.zeros((Mx, My))

    # Compute ang1 (in degrees) in first medium using theta20
    sin_theta20 = math.sin(math.radians(theta20))
    arg = (c1 * sin_theta20) / c2
    if arg > 1:
        arg = 1
    elif arg < -1:
        arg = -1
    ang1 = math.degrees(math.asin(arg))
    
    if math.isinf(DF):
        # Steering-only case
        ux = math.sin(math.radians(ang1)) * math.cos(math.radians(phi)) * math.cos(math.radians(theta)) - \
             math.cos(math.radians(ang1)) * math.sin(math.radians(theta))
        uy = math.sin(math.radians(ang1)) * math.sin(math.radians(phi))
        for i in range(Mx):
            for j in range(My):
                t[i, j] = 1000 * (ux * ex[i] + uy * ey[j]) / c1
        td = np.abs(np.min(t)) + t  # make delays positive
    else:
        # Steering and focusing case
        DQ = DT0 * math.tan(math.radians(ang1)) + DF * math.tan(math.radians(theta20))
        x_val = DQ * math.cos(math.radians(phi))
        y_val = DQ * math.sin(math.radians(phi))
        for i in range(Mx):
            for j in range(My):
                # Calculate distance along the interface for each element
                Db[i, j] = math.sqrt((x_val - ex[i] * math.cos(math.radians(theta)))**2 + (y_val - ey[j])**2)
        # Compute De for each element in x-direction
        De = DT0 + ex * math.sin(math.radians(theta))
        for i in range(Mx):
            for j in range(My):
                xi[i, j] = ferrari2(cr, DF, De[i], Db[i, j])
        for i in range(Mx):
            for j in range(My):
                r1 = math.sqrt(xi[i, j]**2 + De[i]**2)
                r2 = math.sqrt(DF**2 + (Db[i, j] - xi[i, j])**2)
                t[i, j] = 1000 * r1 / c1 + 1000 * r2 / c2
        td = np.max(t) - t

        # Plot ray geometry if requested
        if plt_option.lower() == 'y':
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            for i in range(Mx):
                for j in range(My):
                    xp = np.zeros(3)
                    yp = np.zeros(3)
                    zp = np.zeros(3)
                    xp[0] = ex[i] * math.cos(math.radians(theta))
                    zp[0] = DT0 + ex[i] * math.sin(math.radians(theta))
                    yp[0] = ey[j]
                    xp[1] = ex[i] * math.cos(math.radians(theta)) + xi[i, j] * (x_val - ex[i] * math.cos(math.radians(theta))) / Db[i, j]
                    yp[1] = ey[j] + xi[i, j] * (y_val - ey[j]) / Db[i, j]
                    zp[1] = 0
                    xp[2] = x_val
                    yp[2] = y_val
                    zp[2] = -DF
                    ax.plot(xp, yp, zp, 'b-')
            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Y (mm)")
            ax.set_zlabel("Z (mm)")
            ax.set_title("Ray Geometry - Steering and Focusing Case")
            # Set view: using defaults similar to MATLAB view(25,20)
            ax.view_init(elev=25, azim=20)
            plt.show()

    return td
