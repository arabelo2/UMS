# application/ps_3Dint_service.py

from domain.ps_3Dint import Ps3DInt

def run_ps_3Dint_service(lx, ly, f, mat, ex, ey, angt, Dt0, x, y, z):
    """
    Service layer for computing velocity components using the ps_3Dint algorithm.

    Parameters:
        lx   : float - Length of the array element in the x-direction (mm).
        ly   : float - Length of the array element in the y-direction (mm).
        f    : float - Frequency of the wave (MHz).
        mat  : list  - Material properties [d1, cp1, d2, cp2, cs2, wave_type].
        ex   : float - Offset of the element center from the array center in x (mm).
        ey   : float - Offset of the element center from the array center in y (mm).
        angt : float - Angle of the array relative to the interface (degrees).
        Dt0  : float - Distance from the array center to the interface (mm).
        x    : numpy array - x-coordinates (mm).
        y    : numpy array - y-coordinates (mm).
        z    : numpy array - z-coordinates (mm).

    Returns:
        tuple: (vx, vy, vz) - Velocity components as complex numpy arrays.
    """
    # Input validation
    if lx <= 0 or ly <= 0:
        raise ValueError("lx and ly must be positive.")
    if f <= 0:
        raise ValueError("Frequency f must be positive.")
    if Dt0 < 0:
        raise ValueError("Dt0 (initial delay) cannot be negative.")
    if len(mat) != 6:
        raise ValueError("mat must have 6 elements: [d1, cp1, d2, cp2, cs2, wave_type].")

    # Call the domain logic
    ps = Ps3DInt(lx, ly, f, mat, ex, ey, angt, Dt0)
    return ps.compute_velocity_components(x, y, z)