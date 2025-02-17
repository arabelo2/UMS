# application/mls_array_modeling_service.py

import numpy as np
from domain.mls_array_modeling import MLSArrayModeling

def run_mls_array_modeling_service(f, c, M, dl, gd, Phi, F, wtype, xx, zz):
    """
    Service function to manage the MLS array modeling process.
    
    Parameters:
        f (float): Frequency in MHz.
        c (float): Wave speed in m/s.
        M (int): Number of elements.
        dl (float): Element length divided by wavelength.
        gd (float): Gap size divided by element length.
        Phi (float): Steering angle in degrees.
        F (float): Focal length in mm (np.inf for no focusing).
        wtype (str): Window type ('rect', etc.).
        xx (np.ndarray): X-coordinate mesh.
        zz (np.ndarray): Z-coordinate mesh.
    
    Returns:
        tuple: (p, A, d, g, e) where p is the computed pressure field.
    """
    # Create the MLS array model instance.
    model = MLSArrayModeling(f, c, M, dl, gd, Phi, F, wtype)

    # Calculate array elements (A, d, g, e)
    A, d, g, e = model.calculate_elements()

    # Calculate element pitch s (d + g)
    s = d + g

    # Compute the time delays (in microseconds)
    td = model.calculate_time_delays(s)
    
    # **** FIX: Convert raw delays to phase delays ****
    # In the MATLAB code, we have:
    #     delay = exp(1i * 2 * pi * f * td)
    # f is in MHz and td is in microseconds, so f*td is dimensionless.
    delay = np.exp(1j * 2 * np.pi * f * td)

    # Calculate window amplitudes
    Ct = model.calculate_window_amplitudes()

    # Compute half the element length
    b = d / 2.0

    # Initialize pressure field with the correct shape and complex type.
    p = np.zeros(xx.shape, dtype=complex)
    
    # Sum contributions from each element:
    for mm, centroid in enumerate(e):
        p += Ct[mm] * delay[mm] * model.calculate_pressure_field(b, centroid, xx, zz)
    
    return p, A, d, g, e
