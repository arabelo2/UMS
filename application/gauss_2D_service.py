# application/gauss_2D_service.py

from domain.gauss_2D import gauss_2D

def run_gauss_2D_service(b, f, c, x, z):
    """
    Service function for Gauss_2D calculation.

    Parameters:
        b: float - Half-length of the element (mm)
        f: float - Frequency (MHz)
        c: float - Wave speed in the fluid (m/s)
        x: float or array-like - x-coordinate(s) (mm)
        z: float or array-like - z-coordinate(s) (mm)

    Returns:
        p: complex or array-like - The computed normalized pressure.
    """
    return gauss_2D(b, f, c, x, z)
