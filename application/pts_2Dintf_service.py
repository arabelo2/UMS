# application/pts_2Dintf_service.py

from domain.pts_2Dintf import pts_2Dintf

def run_pts_2Dintf_service(e, xc, angt, Dt0, c1, c2, x, z):
    """
    Service function to run the pts_2Dintf simulation.
    
    Parameters:
        (see pts_2Dintf for details)
    
    Returns:
        xi: The computed intersection point (mm).
    """
    return pts_2Dintf(e, xc, angt, Dt0, c1, c2, x, z)
