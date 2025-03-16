# domain/on_axis_foc2D.py

import numpy as np
import math
from domain.fresnel_int import fresnel_int

def on_axis_foc2D(b, R, f, c, z):
    """
    Compute the on-axis normalized pressure for a 1-D focused piston element 
    of length 2*b (mm) and focal length R (mm) using the paraxial approximation.
    
    The frequency is f (in MHz), b is the transducer half-length (in mm), c is the 
    wave speed (m/s), and z is the on-axis distance (in mm). The propagation phase term 
    exp(ikz) is removed from the calculation.
    
    Parameters:
        b : float
            Transducer half-length (mm).
        R : float
            Focal length (mm).
        f : float
            Frequency (MHz).
        c : float
            Wave speed of the surrounding fluid (m/s).
        z : float or numpy array
            On-axis distance (mm). Can be a scalar or an array.
    
    Returns:
        p : complex or numpy array of complex numbers
            The normalized on-axis pressure.
    
    Procedure:
      1. Ensure no division by zero for z by adding a small epsilon if z==0.
      2. Define the transducer wave number:
             kb = 2000*pi*f*b/c
      3. Compute u = (1 - z/R) and add a small epsilon where u is zero.
      4. Compute the argument for the Fresnel integral and the denominator:
             For elements where z <= R:
                 x_val = sqrt((u*kb*b)/(pi*z))
                 denom_val = sqrt(u)
                 Fr = fresnel_int(x_val)
             For elements where z > R:
                 x_val = sqrt((-u*kb*b)/(pi*z))
                 denom_val = sqrt(-u)
                 Fr = conj(fresnel_int(x_val))
      5. Compute the normalized pressure:
             p = sqrt(2/(1j)) * sqrt((b/R)*kb/pi)   for |u| <= 0.005 (analytical branch)
               + sqrt(2/(1j)) * (Fr/denom_val)        for |u| > 0.005 (numerical branch)
         (Note: The MATLAB code uses element-wise multiplication of these branches.)
    
    All operations are vectorized so that z can be a scalar or an array.
    """
    # Ensure z is a numpy array for element-wise operations.
    z = np.array(z, dtype=float)
    eps_val = np.finfo(float).eps

    # Avoid division by zero: if z==0, add a small epsilon.
    z = z + eps_val * (z == 0)

    # Compute the transducer wave number.
    kb = 2000 * math.pi * f * b / c

    # Compute u = (1 - z/R) and ensure u is not zero.
    u = 1 - z / R
    u = u + eps_val * (u == 0)

    # Compute the argument for the Fresnel integral.
    # For z <= R, use positive u; for z > R, use negative u.
    # We use element-wise conditions.
    x_val = np.where(z <= R,
                     np.sqrt((u * kb * b) / (np.pi * z)),  # Allow complex results
                     np.sqrt((-u * kb * b) / (np.pi * z)))  # Allow complex results
    
    # Denom is computed similarly.
    denom_val = np.where(z <= R,
                         np.sqrt(u),  # Allow complex results
                         np.sqrt(-u))  # Allow complex results
    
    # Compute the Fresnel integral: for z<=R use fresnel_int(x_val), for z>R use its conjugate.
    Fr = np.where(z <= R,
                  fresnel_int(x_val),
                  np.conjugate(fresnel_int(x_val)))
    
    # Compute the scaling factor sqrt(2/(1j))
    scaling = np.sqrt(2 / (1j))
    analytic_value = scaling * np.sqrt((b / R) * kb / math.pi)
    
    # Choose branch based on |u|: if |u| <= 0.005 use analytic, else numerical.
    p = np.where(np.abs(u) <= 0.005, analytic_value, scaling * (Fr / denom_val))
    
    return p
