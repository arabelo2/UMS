#!/usr/bin/env python3
"""
Service module: delay_laws3Dint_service.py

Provides a service to compute delay laws for a 2D array through a planar interface.
"""

import numpy as np
from domain.delay_laws3Dint import delay_laws3Dint  # Import your domain function

def run_delay_laws3Dint_service(Mx: int, My: int, sx: float, sy: float,
                                theta: float, phi: float, theta20: float,
                                DT0: float, DF: float, c1: float, c2: float,
                                plt_option: str = 'n',
                                view_elev: float = 25.0, view_azim: float = 20.0,
                                z_scale: float = 1.0) -> np.ndarray:
    """
    Compute delay laws (in microseconds) for a 2D array by calling the domain function,
    then applies an optional scaling factor (z_scale) to the computed delays.
    
    Parameters:
        Mx, My    : Number of elements in the x and y directions.
        sx, sy    : Element pitches in the x and y directions (mm).
        theta     : Array angle with the interface (degrees).
        phi       : Steering angle for the second medium (degrees).
        theta20   : Refracted steering angle in the second medium (degrees).
        DT0       : Height of the array center above the interface (mm).
        DF        : Focal distance in the second medium (mm); use inf for steering-only.
        c1, c2    : Wave speeds (m/s) in the first and second media.
        plt_option: Plot option ('y' or 'n').
        view_elev : Camera elevation for 3D plot.
        view_azim : Camera azimuth for 3D plot.
        z_scale   : Scaling factor to be applied to the delays. Default: 1.0.
        
    Returns:
        A 2D NumPy array of computed time delays (in microseconds) scaled by z_scale.
    """
    # Compute delay laws using the domain function.
    td = delay_laws3Dint(Mx, My, sx, sy, theta, phi, theta20, DT0, DF, c1, c2, plt_option, view_elev, view_azim)
    td_scaled = td * z_scale
    return td_scaled
