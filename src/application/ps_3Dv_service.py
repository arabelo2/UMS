# application/ps_3Dv_service.py

from domain.ps_3Dv import RectangularPiston3D

def run_ps_3Dv_service(lx: float, ly: float, f: float, c: float, ex: float, ey: float, x, y, z, P: int = None, Q: int = None):
    """
    Service function to compute the normalized pressure from a rectangular piston.

    Parameters:
        lx, ly (float): Element dimensions along x and y (mm).
        f (float): Frequency (MHz).
        c (float): Wave speed (m/s).
        ex, ey (float): Lateral offsets (mm).
        x, y, z: Evaluation coordinates (scalars or NumPy arrays, in mm).
        P, Q (int, optional): Number of segments for integration along x and y.

    Returns:
        Computed normalized pressure (complex scalar or array).
    """
    piston = RectangularPiston3D(lx, ly, f, c, ex, ey)
    return piston.compute_pressure(x, y, z, P, Q)
