# application/on_axis_foc2D_service.py

from domain.on_axis_foc2D import on_axis_foc2D

def run_on_axis_foc2D_service(b, R, f, c, z):
    """
    Service function for on_axis_foc2D.
    
    Parameters:
        b : float
            Transducer half-length (mm).
        R : float
            Focal length (mm).
        f : float
            Frequency (MHz).
        c : float
            Wave speed of the surrounding fluid (m/s).
        z : float or numpy array
            On-axis distance (mm) at which to compute the pressure.
            
    Returns:
        p : complex or numpy array of complex numbers
            The normalized on-axis pressure computed by the domain function.
    """
    return on_axis_foc2D(b, R, f, c, z)
