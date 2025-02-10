# domain/pts_2Dintf.py

import numpy as np
from domain.ferrari2 import ferrari2

def pts_2Dintf(e, xc, angt, Dt0, c1, c2, x, z):
    """
    Compute the intersection distance xi (in mm) along the interface where the ray from
    the center of a segment (with offset xc) to the observation point (x,z) in the second medium
    intersects the interface.
    
    Parameters:
        e    : float
               Offset of the element center from the array center (mm).
        xc   : float
               Offset of the segment from the element center (mm).
        angt : float
               Angle of the array relative to the x-axis (degrees).
        Dt0  : float
               Distance from the array center to the interface (mm).
        c1   : float
               Wave speed in medium one (m/s).
        c2   : float
               Wave speed in medium two (m/s).
        x    : float
               x-coordinate of the observation point in medium two (mm).
        z    : float
               z-coordinate of the observation point in medium two (mm).
    
    Returns:
        xi   : float
               The intersection point along the interface (in mm).
               
    Procedure:
        1. Compute Dtn, the effective vertical distance from the array center to the interface:
             Dtn = Dt0 + (e + xc) * sin(angt in radians)
        2. Compute Dxn, the effective horizontal distance from the segment to the observation point:
             Dxn = x - (e + xc) * cos(angt in radians)
        3. Set:
             cr = c1 / c2,
             DF = z,
             DT = Dtn,
             DX = Dxn.
        4. Call ferrari2(cr, DF, DT, DX) to obtain xi.
    """
    # Convert angle to radians
    angt_rad = np.deg2rad(angt)
    
    # Compute effective vertical and horizontal distances.
    Dtn = Dt0 + (e + xc) * np.sin(angt_rad)
    Dxn = x - (e + xc) * np.cos(angt_rad)
    
    # Wave speed ratio.
    cr = c1 / c2
    
    # Set parameters for ferrari2.
    DF = z    # Depth in medium two (assumed positive)
    DT = Dtn  # Effective height in medium one.
    DX = Dxn  # Horizontal distance along the interface.
    
    xi = ferrari2(cr, DF, DT, DX)
    return xi
