# application/fresnel_2D_service.py

from domain.fresnel_2D import fresnel_2D

def run_fresnel_2D_service(b, f, c, x, z):
    """
    Service function to compute the Fresnel-based pressure field for a 1-D element.
    
    Parameters:
      b : float
          Half-length of the element (mm).
      f : float
          Frequency (MHz).
      c : float
          Wave speed (m/s).
      x : float or array-like
          x-coordinate(s) in mm.
      z : float or array-like
          z-coordinate(s) in mm.
    
    Returns:
      p : complex or numpy array of complex numbers
          The normalized pressure computed by fresnel_2D.
    """
    return fresnel_2D(b, f, c, x, z)
