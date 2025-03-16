# domain/mls_array_modeling_gauss.py

import numpy as np
from domain.elements import ElementsCalculator

# For delay_laws2D: try service first, then domain fallback.
try:
    from application.delay_laws2D_service import delay_laws2D as delay_laws2D_service
except ImportError:
    from domain.delay_laws2D import delay_laws2D as delay_laws2D_service

# For discrete_windows: try service first, then domain fallback.
try:
    from application.discrete_windows_service import discrete_windows as discrete_windows_service
except ImportError:
    from domain.discrete_windows import discrete_windows as discrete_windows_service

# For NPGauss_2D: try service first, then domain fallback.
try:
    from application.NPGauss_2D_service import run_np_gauss_2D_service as NPGauss_2D_service
except ImportError:
    from domain.NPGauss_2D import np_gauss_2D as NPGauss_2D_service

def compute_pressure_field(f, c, M, dl, gd, Phi, F, window_type):
    """
    Compute the normalized pressure field for the Gaussian MLS array modeling.
    
    Parameters:
        f (float): Frequency in MHz.
        c (float): Wave speed in m/sec.
        M (int): Number of array elements.
        dl (float): Diameter ratio (normalized element length).
        gd (float): Gap ratio (normalized gap between elements).
        Phi (float): Steering angle in degrees.
        F (float): Focal length in mm (use float('inf') for no focusing).
        window_type (str): Type of amplitude weighting function.
    
    Returns:
        dict: A dictionary containing:
            - 'p': The computed pressure field (2D NumPy array, complex).
            - 'x': x-coordinates of the field (1D NumPy array).
            - 'z': z-coordinates of the field (1D NumPy array).
    """
    # Calculate array geometry using ElementsCalculator
    calc = ElementsCalculator(frequency_mhz=f, wave_speed_m_s=c, 
                                diameter_ratio=dl, gap_ratio=gd, num_elements=M)
    A, d, g, centroids = calc.calculate()  # centroids now represent element positions (e)
    b = d / 2.0
    s = d + g

    # Generate spatial grid for field calculations
    z = np.linspace(1, 100 * dl, 500)
    x = np.linspace(-50 * dl, 50 * dl, 500)
    xx, zz = np.meshgrid(x, z)

    # Generate time delays using the delay_laws2D service
    td = delay_laws2D_service(M, s, Phi, F, c)
    delay = np.exp(1j * 2 * np.pi * f * td)

    # Retrieve amplitude weights using the discrete_windows service
    Ct = discrete_windows_service(M, window_type)

    # Calculate the normalized pressure field by summing contributions from each element
    p = np.zeros_like(xx, dtype=complex)
    for mm in range(M):        
        p += Ct[mm] * delay[mm] * NPGauss_2D_service(b, f, c, centroids[mm], xx, zz)
        
    return {'p': p, 'x': x, 'z': z}
