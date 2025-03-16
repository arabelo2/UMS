# application/mls_array_modeling_gauss_service.py

from domain.mls_array_modeling_gauss import compute_pressure_field

def run_mls_array_modeling_gauss(f, c, M, dl, gd, Phi, F, window_type):
    """
    Service layer for the Gaussian MLS array modeling.
    Validates input parameters and calls the domain function.

    Parameters:
        f (float): Frequency (MHz).
        c (float): Wave speed (m/sec).
        M (int): Number of elements.
        dl (float): Normalized element length.
        gd (float): Normalized gap between elements.
        Phi (float): Steering angle (degrees).
        F (float): Focal length (mm), or float('inf') for no focusing.
        window_type (str): Type of amplitude weighting function.
        
    Returns:
        dict: Dictionary with the pressure field ('p'), x-coordinates ('x'), and z-coordinates ('z').
    """
    # (Input validation could be added here if needed)
    return compute_pressure_field(f, c, M, dl, gd, Phi, F, window_type)
