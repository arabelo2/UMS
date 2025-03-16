# application/pts_3Dintf_service.py

from domain.pts_3Dintf import Pts3DIntf

def run_pts_3Dintf_service(ex, ey, xn, yn, angt, Dt0, c1, c2, x, y, z):
    """
    Computes the intersection point xi based on the given parameters.

    Parameters:
        ex, ey, xn, yn, angt, Dt0, c1, c2, x, y, z - Various numerical inputs.

    Returns:
        xi: Computed intersection point.
    """
    # Input validation
    if Dt0 < 0:
        raise ValueError("Dt0 (initial delay) cannot be negative.")
    if c1 <= 0 or c2 <= 0:
        raise ValueError("Wave speeds c1 and c2 must be positive.")

    # Call the domain logic
    pts = Pts3DIntf(ex, ey, xn, yn, angt, Dt0, c1, c2)
    return pts.compute_intersection(x, y, z)
