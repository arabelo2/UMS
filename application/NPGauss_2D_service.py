# application/NPGauss_2D_service.py

from domain.NPGauss_2D import np_gauss_2D

def run_np_gauss_2D_service(b, f, c, e, x, z):
    """
    Service function for NPGauss_2D calculation.

    Parameters:
        b (float): Half-length of the element (mm)
        f (float): Frequency (MHz)
        c (float): Wave speed in the fluid (m/s)
        e (float): Offset of the element's center in the x-direction (mm)
        x (float or array-like): x-coordinate(s) (mm)
        z (float or array-like): z-coordinate(s) (mm)

    Returns:
        p (np.ndarray): The computed normalized pressure field (complex array)
    """
    return np_gauss_2D(b, f, c, e, x, z)
