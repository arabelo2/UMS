# domain/fresnel_2D.py

import numpy as np
import math
from domain.fresnel_int import fresnel_int

def fresnel_2D(b, f, c, x, z):
    """
    Compute the normalized pressure field at a point or points (x, z) (in mm)
    for a 1-D element of length 2*b (in mm) radiating into a fluid with
    wave speed c (m/s) at frequency f (MHz). This function uses fresnel_int
    to evaluate the Fresnel integral.

    Parameters:
      b : float
          Half-length of the element (mm).
      f : float
          Frequency (MHz).
      c : float
          Wave speed in the fluid (m/s).
      x : float or numpy array
          x-coordinate(s) (mm).
      z : float or numpy array
          z-coordinate(s) (mm). Typically a scalar, but can be an array.

    Returns:
      p : complex or numpy array of complex numbers
          The normalized pressure.

    Procedure:
      1. Compute wave number: kb = 2000 * pi * f * b / c
      2. Normalize x and z by b: xb = x / b, zb = z / b
      3. Compute argument for the Fresnel integral:
            arg = sqrt(kb / (pi * zb))   (with a small epsilon if zb=0)
      4. Evaluate Fresnel integrals for (xb+1)*arg and (xb-1)*arg
      5. Combine results:
            factor = sqrt(1/(2i))
            p = factor * exp(i*kb*zb) * [fresnel_int(arg*(xb+1)) - fresnel_int(arg*(xb-1))]
    """
    kb = 2000 * math.pi * f * b / c
    
    # Convert x, z to numpy arrays for vectorized operations.
    xb = np.array(x, dtype=float) / b
    zb = np.array(z, dtype=float) / b

    # Avoid division by zero if zb=0.
    epsilon = 1e-12
    zb = np.where(zb == 0, epsilon, zb)

    arg = np.sqrt(kb / (math.pi * zb))

    I1 = fresnel_int(arg * (xb + 1))
    I2 = fresnel_int(arg * (xb - 1))

    factor = np.sqrt(1 / (2j))
    p = factor * np.exp(1j * kb * zb) * (I1 - I2)
    return p
