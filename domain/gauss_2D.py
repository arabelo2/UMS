# domain/gauss_2D.py

import numpy as np
from domain.gauss_c15 import gauss_c15

def gauss_2D(b, f, c, x, z):
    """
    Calculate the normalized pressure field for a 1-D element using a multi-Gaussian beam model.

    Parameters:
        b: float - Half-length of the element (mm)
        f: float - Frequency (MHz)
        c: float - Wave speed in the fluid (m/s)
        x: np.ndarray or float - x-coordinate(s) (mm)
        z: np.ndarray or float - z-coordinate(s) (mm)

    Returns:
        np.ndarray - The computed normalized pressure field
    """
    # Retrieve coefficients
    A, B = gauss_c15()

    # Calculate wave number
    kb = 2000 * np.pi * f * b / c

    # Normalize coordinates
    xb = x / b
    zb = z / b

    # Initialize pressure field
    p = np.zeros_like(xb, dtype=complex)

    # Superimpose 15 Gaussian beams
    for nn in range(15):
        qb = zb - 1j * 1000 * np.pi * f * b / (B[nn] * c)
        qb0 = -1j * 1000 * np.pi * f * b / (B[nn] * c)
        p += np.sqrt(qb0 / qb) * A[nn] * np.exp(1j * kb * xb**2 / (2 * qb))

    return p
