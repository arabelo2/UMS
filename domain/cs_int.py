# domain/cs_int.py

import numpy as np

def cs_int(xi):
    """
    Compute approximate cosine and sine integrals for positive values of xi.
    
    The approximations are based on the expressions given in Abramowitz and Stegun,
    Handbook of Mathematical Functions (Dover Publications, 1965, pp. 301-302).
    
    Parameters:
        xi : float or numpy array
            Input value(s) for which the cosine and sine integrals are computed.
            Should be non-negative.
            
    Returns:
        c : float or numpy array
            Approximation of the cosine integral.
        s : float or numpy array
            Approximation of the sine integral.
    
    The approximations are computed as follows:
    
        f = (1 + 0.926 * xi) / (2 + 1.792 * xi + 3.104 * xi**2)
        g = 1 / (2 + 4.142 * xi + 3.492 * xi**2 + 6.67 * xi**3)
        c = 0.5 + f * sin(pi * xi**2 / 2) - g * cos(pi * xi**2 / 2)
        s = 0.5 - f * cos(pi * xi**2 / 2) - g * sin(pi * xi**2 / 2)
    
    Both c and s are returned as floats or numpy arrays, matching the input type.
    """
    # Ensure xi is a numpy array for element-wise operations.
    xi = np.array(xi, dtype=float)
    
    f_val = (1 + 0.926 * xi) / (2 + 1.792 * xi + 3.104 * xi**2)
    g_val = 1 / (2 + 4.142 * xi + 3.492 * xi**2 + 6.67 * xi**3)
    
    c = 0.5 + f_val * np.sin(np.pi * xi**2 / 2) - g_val * np.cos(np.pi * xi**2 / 2)
    s = 0.5 - f_val * np.cos(np.pi * xi**2 / 2) - g_val * np.sin(np.pi * xi**2 / 2)
    
    return c, s
