# domain/interface2.py

import numpy as np

def interface2(x, cr, df, dp, dpf):
    """
    Compute the value of the function y that is zero when Snell's law is satisfied.
    
    Parameters:
        x   : float or numpy array
              Candidate location(s) along the interface (in mm).
        cr  : float
              Ratio c1/c2, where c1 is the wave speed in medium one and c2 is in medium two.
        df  : float
              Depth of the point in medium two (DF, in mm).
        dp  : float
              Height of the point in medium one (DT, in mm).
        dpf : float
              Separation distance between the points in medium one and two (DX, in mm).
    
    Returns:
        y   : float or numpy array
              The computed function value; when y = 0, Snell's law is satisfied.
              
    The function computes:
        y = x / sqrt(x^2 + dp^2) - cr * (dpf - x) / sqrt((dpf - x)^2 + df^2)
    """
    # Convert x to a NumPy array to support element-wise operations
    x = np.array(x, dtype=float)
    
    term1 = x / np.sqrt(x**2 + dp**2)
    denom = np.sqrt((dpf - x)**2 + df**2)
    denom = np.where(denom == 0, 1e-9, denom)  # Avoid division by zero
    term2 = cr * (dpf - x) / denom

    y = term1 - term2
    return y
