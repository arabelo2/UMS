# domain/delay_laws2D.py

import numpy as np
import math

def delay_laws2D(M: int,
                 s: float,
                 Phi: float,
                 F: float,
                 c: float) -> np.ndarray:
    """
    Calculate the time delays (in microseconds) for an M-element array
    with pitch s (mm), steering angle Phi (degrees), focus distance F (mm),
    and wave speed c (m/s). Steering only is supported by setting F to np.inf.

    Parameters:
        M (int)    : Number of elements in the array (assumed odd, but must be >= 1).
        s (float)  : Pitch of the array in mm.
        Phi (float): Steering angle in degrees.
        F (float)  : Focal distance in mm (set to np.inf for steering only).
        c (float)  : Wave speed (m/s).

    Returns:
        np.ndarray : A 1D NumPy array of length M containing the time delays
                     (in microseconds) for each element.

    Notes:
    - Based on the MATLAB code:
        td = delay_laws2D(M,s,Phi,F,c)
      which calculates time delay in microseconds.
    - Steering only:
        if F == inf, use one of two formulas depending on sign of Phi.
    - Steering + focusing:
        compute distances rm for each element, and reference distance r1 or rM,
        then compute time difference.

    Caveats:
    - M must be >= 1. If M < 1, raises ValueError.
    - c must be nonzero, otherwise raises ValueError for division by zero.
    """

    # Ensure M >= 1
    if M < 1:
        raise ValueError("Number of elements M must be >= 1.")

    # If wave speed is 0, we will get division by zero:
    if c == 0:
        raise ValueError("Wave speed c cannot be zero (division by zero).")

    # M must be an odd integer for symmetrical array indexing:
    # but logic can still run if M is even or M=1, just no symmetrical center logic
    Mb = (M - 1) / 2.0

    # element indices: 1...M
    m = np.arange(1, M + 1)

    # location of centroids of each element
    em = s * ((m - 1) - Mb)

    # initialize time_delay array
    td = np.zeros(M, dtype=float)

    # Steering only case
    if math.isinf(F):
        if Phi > 0:
            # td=1000*s*sind(Phi)*(m-1)/c
            td = (1000.0 * s * math.sin(math.radians(Phi)) * (m - 1)) / c
        else:
            # td=1000*s*sind(abs(Phi))*(M-m)/c
            td = (1000.0 * s * math.sin(math.radians(abs(Phi))) * (M - m)) / c
    else:
        # focusing and steering
        r1 = math.sqrt(F**2 + (Mb * s)**2 + 2 * F * Mb * s * math.sin(math.radians(Phi)))
        rm = np.sqrt(F**2 + em**2 - 2 * F * em * math.sin(math.radians(Phi)))
        rM = math.sqrt(F**2 + (Mb * s)**2 + 2 * F * Mb * s * math.sin(math.radians(abs(Phi))))

        if Phi > 0:
            td = 1000.0 * (r1 - rm) / c
        else:
            td = 1000.0 * (rM - rm) / c

    return td
