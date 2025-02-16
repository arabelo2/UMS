# domain/NPGauss_2D.py

import numpy as np
import math
from domain.gauss_c10 import gauss_c10

def np_gauss_2D(b: float,
                f: float,
                c: float,
                e: float,
                x,
                z):
    """
    Calculate the normalized pressure field for a 1-D element using a non-paraxial
    multi-Gaussian beam model (NPGauss_2D).

    Parameters:
      b : float
          Half-length of the element (mm).
      f : float
          Frequency (MHz).
      c : float
          Wave speed in the fluid (m/s).
      e : float
          Offset of the element's center in the x-direction (mm).
      x : float or numpy.ndarray
          x-coordinate(s) (mm) where pressure is computed.
      z : float or numpy.ndarray
          z-coordinate(s) (mm) where pressure is computed.

    Returns:
      np.ndarray
          The computed normalized pressure field as a complex NumPy array.

    Notes:
      - This function uses Wen & Breazeale's 10-coefficient multi-Gaussian model (gauss_c10).
      - The wave number is computed as: kb = 2000 * pi * f * b / c.
      - If c == 0, raises ValueError to prevent division by zero.
      - The domain logic replicates the original NPGauss_2D.m MATLAB function.

    Reference:
      Wen, J.J. and M. A. Breazeale, 
      "Computer optimization of the Gaussian beam description of an ultrasonic field,"
      Computational Acoustics, Vol.2, D. Lee, A. Cakmak, R. Vichnevetsky, Eds.,
      Elsevier Science Publishers, Amsterdam, 1990, pp. 181-196.
    """

    if c == 0:
        raise ValueError("Wave speed c cannot be zero (division by zero).")

    # Retrieve the 10 Gaussian coefficients
    A, B = gauss_c10()

    # Compute wave number
    kb = 2000.0 * math.pi * f * b / c

    # Normalize coordinates
    xb = x / b
    zb = z / b
    eb_norm = e / b

    # Distance from offset center
    Rb = np.sqrt((xb - eb_norm)**2 + (zb)**2)

    # DB = kb/2
    Db = kb / 2.0

    # cosp = zb./Rb => cos(phase?), must handle Rb=0 if that arises
    # We'll assume user doesn't pass exactly Rb=0. If needed, could add small epsilon.
    cosp = zb / Rb

    # Initialize pressure field
    p = np.zeros_like(xb, dtype=complex)

    # Accumulate contributions from the 10 Gaussians
    for nn in range(10):
        arg = cosp**2 + 1j * B[nn] * Rb / Db
        Dn = np.sqrt(arg)
        amp = A[nn] * np.exp(1j * kb * Rb) / Dn
        p += amp * np.exp(-1j * kb * (xb**2) / (2.0 * Rb * arg))

    return p
