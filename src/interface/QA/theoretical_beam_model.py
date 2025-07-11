#!/usr/bin/env python3
# src/interface/theoretical_beam_model.py

import numpy as np
from scipy.optimize import fsolve
from scipy.signal import hilbert # Used for envelope in analytical models sometimes, but np.abs for complex is simpler here
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Adjust path to import fwhm_methods_addon.py
# Assuming src/interface/QA is added to sys.path in the main script,
# or a relative import that works from the project root.
# For simplicity in this example, we'll try a relative import from 'domain' to 'interface/QA'
# or assume fwhm_methods_addon is directly accessible.
# A more robust way for a shared utility is to place fwhm_methods_addon in 'application/utils' or similar.

# For this example, let's assume fwhm_methods_addon.py is added to the Python path
# or handled by the calling script's path adjustments.
# For local testing, you might need to manually add:
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'interface', 'QA')))
from fwhm_methods_addon import estimate_fwhm_F2 ### CHANGE: Import F2 instead of F1


# --- Helper Function for Numerical Solving (find_interface_intersection_point) ---
def _ray_path_equation(x_interface, source_x, target_x, d1, z_target, c1, c_solid_focus):
    """
    Function to find the root for the interface intersection point (x_interface).
    This equation ensures Snell's Law and geometric alignment for the ray path.
    Based on constant horizontal slowness across interface.
    """
    # Vector from source to interface point
    vec1_x = x_interface - source_x
    vec1_z = d1

    # Vector from interface point to target point
    vec2_x = target_x - x_interface
    vec2_z = z_target - d1

    # Avoid division by zero if target_z == d1 or source_z == d1 (which it is)
    # Ensure dist_fluid and dist_solid are positive
    dist_fluid = np.sqrt(vec1_x**2 + vec1_z**2)
    dist_solid = np.sqrt(vec2_x**2 + vec2_z**2)

    # Condition for refraction: horizontal slowness is conserved
    # (vec1_x / (dist_fluid * c1)) - (vec2_x / (dist_solid * c_solid_focus)) = 0
    
    # Handle cases where path lengths might be zero or near zero to prevent NaNs/Infs
    if dist_fluid < 1e-9: dist_fluid = 1e-9 # Small epsilon to prevent div by zero
    if dist_solid < 1e-9: dist_solid = 1e-9

    return (vec1_x / (dist_fluid * c1)) - (vec2_x / (dist_solid * c_solid_focus))


def _calculate_single_ray_contribution(element_coords, target_coords, params, c_solid_focus):
    """
    Calculates the time-of-flight (TOF) and complex amplitude for a single ray
    from one transducer element to a target point in the solid, considering the interface.
    """
    source_x, source_y, source_z = element_coords
    target_x, target_y, target_z = target_coords
    d1 = params['d1']  # Water path thickness
    c1 = params['c1']  # Speed of sound in fluid
    f = params['f'] * 1e6 # Frequency in Hz

    # Initial guess for x_interface (linear interpolation from source to target at interface depth)
    # Handle case where target_z is at d1 (interface) or source_z is 0 and target_z is d1
    if abs(target_z - source_z) < 1e-9: # Ray is purely horizontal, or target is on transducer plane
        # This case requires special handling for vertical incidence or horizontal rays.
        # For vertical incidence, x_interface = source_x.
        # For horizontal rays, no refraction, but this scenario is unlikely for focusing.
        x_interface_guess = source_x
    else:
        x_interface_guess = source_x + (target_x - source_x) * (d1 - source_z) / (target_z - source_z)

    # Solve for x_interface using fsolve
    try:
        # We need to solve for x_interface for the ray in the X-Z plane.
        # The y-coordinate on the interface is a linear interpolation, no refraction in Y-Z plane if interface is planar.
        # However, for a general 3D ray tracing, if target_y is non-zero, y_interface will not be source_y.
        # y_interface = source_y + (target_y - source_y) * (d1 - source_z) / (target_z - source_z)

        # For the 2D FWHM, we're assuming y_vec = 0, so y_interface = 0, y_target = 0
        x_interface_result = fsolve(_ray_path_equation, x_interface_guess,
                                    args=(source_x, target_x, d1, target_z, c1, c_solid_focus),
                                    factor=0.1, epsfcn=1e-8, xtol=1e-6) # Added xtol for convergence
        x_interface = x_interface_result[0]
        y_interface = source_y # Assuming y=0 plane for FWHM analysis
        
    except Exception as e:
        # print(f"Warning: fsolve failed for ray from {element_coords} to {target_coords}. Error: {e}")
        return np.inf, 0.0 + 0.0j # Return large TOF, zero amplitude for failed ray

    interface_coords = (x_interface, y_interface, d1)

    # Path length in fluid
    L1 = np.sqrt(np.sum((np.array(interface_coords) - np.array(element_coords))**2))

    # Path length in solid
    L2 = np.sqrt(np.sum((np.array(target_coords) - np.array(interface_coords))**2))

    if L1 < 1e-9 or L2 < 1e-9: # Avoid division by zero for amplitude
        return np.inf, 0.0 + 0.0j

    time_of_flight = L1 / c1 + L2 / c_solid_focus

    # Simple amplitude decay (1/R for pressure amplitude)
    amplitude = 1.0 / (L1 + L2)

    complex_amplitude = amplitude * np.exp(1j * 2 * np.pi * f * time_of_flight)

    return time_of_flight, complex_amplitude


def calculate_focal_depth_in_solid(params):
    """
    Calculates the theoretical focal depth in the solid medium based on the F-number,
    transducer geometry, and fluid-solid interface using a simplified refraction model.
    This is an approximation for an ideal focus.

    Args:
        params (dict): Dictionary containing simulation parameters like Dt0, DF, d1, c1, c2, cs2, wave_type.

    Returns:
        float: The theoretical focal depth in the solid (mm).
    """
    Dt0 = params['Dt0'] # Total transducer width (mm)
    DF = params['DF'] # F-number
    d1 = params['d1'] # Water path (thickness of fluid layer)
    c1 = params['c1'] # Water speed [m/s]
    c2 = params['c2'] # Steel Cp speed [m/s]
    cs2 = params['cs2'] # Steel Cs speed [m/s]
    wave_type = params['wave_type'] # 'p' or 's'

    c_solid_focus = c2 if wave_type == 'p' else cs2

    if DF == float('inf'):
        print("Warning: DF is 'inf'. Theoretical focal depth not well-defined. Returning a very large number.")
        return 1e6 # Represents an unfocused beam in the far field

    # Nominal focal length in water based on F-number
    F_water_nominal = Dt0 * DF

    # --- Theoretical Focal Depth Calculation using Snell's Law and Geometric Focus ---
    # This involves finding the intersection point of refracted rays.
    # For a simple planar interface and a flat transducer, the focal shift is often approximated.
    # A common formula for focal depth 'F_solid' after a planar interface:
    # F_solid = d1 + (F_water_nominal - d1) * (c_solid_focus / c1)
    # This formula is generally valid for near-axial rays or when the focus is deep in the second medium.
    
    if F_water_nominal <= d1:
        # If the nominal focus in water is at or before the interface, the beam
        # enters the solid diverging or parallel. A distinct 'focal point'
        # in the solid may not exist or be well-defined by this simple formula.
        # For simplicity, we can set the focal depth to just after the interface or use a heuristic.
        print(f"Warning: Nominal focal length in water ({F_water_nominal:.2f} mm) is <= water path ({d1:.2f} mm). "
              "Beam might be diverging in solid. Theoretical focus might be inaccurate.")
        # Heuristic: if focus is in water, use the interface plus some small offset,
        # or propagate based on the last divergence from water to solid
        # Let's use the interface plus 10mm as a placeholder if focus is *in* water, otherwise use normal formula.
        if F_water_nominal < 1.0: # Very shallow focus
            b_focal_theoretical = d1 + (Dt0 / 2.0) # Arbitrary small spread after interface
        else:
             b_focal_theoretical = d1 + (F_water_nominal / d1) * c_solid_focus # Ratio-based heuristic

    else:
        # Normal case: focus is in the solid
        b_focal_theoretical = d1 + (F_water_nominal - d1) * (c_solid_focus / c1)

    return b_focal_theoretical


# --- Main Function for Theoretical FWHM Calculation ---
def calculate_theoretical_fwhm_ray_acoustics(params):
    """
    Calculates the theoretical FWHM of the ultrasonic beam through a fluid-solid
    interface using a ray acoustics simulation, accounting for refraction and mode conversion.

    Based on principles from Xu & O'Reilly (2020) [10.1109/TBME.2019.2912146].
    FWHM extraction based on Rainio et al. (2025) [2025-02 Methods for estimating full width at half maximum.pdf].

    Args:
        params (dict): Dictionary containing all simulation parameters.

    Returns:
        tuple: (theoretical_fwhm_value, theoretical_beam_profile)
               theoretical_fwhm_value (float): The calculated theoretical FWHM in mm.
               theoretical_beam_profile (np.ndarray): The theoretical beam profile (envelope) for plotting.
    """
    # --- Parameter Extraction ---
    c1 = params['c1'] # Speed of sound in fluid (water) [m/s]
    c2 = params['c2'] # Longitudinal speed of sound in solid (steel) [m/s]
    cs2 = params['cs2'] # Shear speed of sound in solid (steel) [m/s]
    d1 = params['d1'] # Thickness of fluid layer (mm)
    f = params['f'] * 1e6 # Frequency (Hz)
    wave_type = params['wave_type'] # 'p' for longitudinal, 's' for shear

    c_solid_focus = c2 if wave_type == 'p' else cs2
    
    # Calculate the theoretical focal depth in the solid
    b_focal_theoretical = calculate_focal_depth_in_solid(params)
    
    if b_focal_theoretical is None:
        print("Theoretical focal depth could not be determined. Aborting theoretical FWHM calculation.")
        return 0.0, np.array([]) # Return default empty values

    # --- Define Theoretical Scan Grid (centered on calculated focal depth) ---
    # Use parameters['xs'] and parameters['zs'] for defining the bounds
    # Assuming params['xs'] and params['zs'] are already parsed numpy arrays or lists (as handled in plot_pipeline_results.py)
    x_start = params['xs'].min()
    x_end = params['xs'].max()
    z_start = params['zs'].min()
    z_end = params['zs'].max()

    # Use higher resolution for theoretical accuracy, but within bounds of original scan
    theoretical_x_num_points = int(len(params['xs']) * 5) # 5x resolution
    # For Z, we primarily care about the focal plane, so a smaller range around focal_depth is sufficient
    # but the full theoretical_field still needs a sensible Z range for summation.
    theoretical_z_num_points = int(len(params['zs']) * 2) # 2x resolution along Z

    x_scan_range = np.linspace(x_start, x_end, theoretical_x_num_points)
    
    # Define z_scan_range for the theoretical field (around the calculated focal depth)
    z_margin = max(5.0, (z_end - z_start) / 4.0) # At least 5mm margin or 1/4th of original Z scan range
    z_scan_range = np.linspace(b_focal_theoretical - z_margin,
                               b_focal_theoretical + z_margin,
                               theoretical_z_num_points)

    theoretical_field = np.zeros((len(z_scan_range), len(x_scan_range)), dtype=complex)

    # --- Transducer Elements Definition ---
    L1 = params['L1'] # Elements in X
    L2 = params['L2'] # Elements in Y
    lx = params['lx'] # Element width X
    gx = params['gx'] # Kerf X
    ly = params['ly'] # Element height Y # Not strictly used for 2D X-Z FWHM, but useful for aperture
    gy = params['gy'] # Kerf Y # Not strictly used for 2D X-Z FWHM

    # Calculate actual element centers (assuming array centered at 0,0,0)
    element_centers_x = np.linspace(
        -((L1 - 1) / 2.0) * (lx + gx),
        ((L1 - 1) / 2.0) * (lx + gx),
        L1
    )
    element_centers_y = np.linspace(
        -((L2 - 1) / 2.0) * (ly + gy),
        ((L2 - 1) / 2.0) * (ly + gy),
        L2
    )
    # If L2 is 1 and y_vec is 0, element_centers_y will just contain 0.0

    # --- Ray Summation Loop ---
    # Assuming y_vec is a single value, and we are analyzing the beam in the Y=0 plane.
    y_target_plane = float(params.get('y_vec', 0.0))
    if isinstance(y_target_plane, np.ndarray) and y_target_plane.size > 0:
        y_target_plane = y_target_plane[0] # Use the first Y position if y_vec is an array

    for elem_x in element_centers_x:
        for elem_y in element_centers_y: # Even if L2=1, this loop runs once for elem_y = 0.0
            element_coords = (elem_x, elem_y, 0.0) # Transducer is at Z=0

            for i_z, z_target in enumerate(z_scan_range):
                for i_x, x_target in enumerate(x_scan_range):
                    target_coords = (x_target, y_target_plane, z_target) # Use the y_target_plane
                    
                    # Calculate contribution for this element-target pair
                    tof, complex_amp = _calculate_single_ray_contribution(
                        element_coords, target_coords, params, c_solid_focus
                    )
                    
                    if np.isinf(tof): # If fsolve failed or path invalid
                        continue # Skip this contribution
                    
                    # Sum coherent contributions
                    theoretical_field[i_z, i_x] += complex_amp

    # --- Post-processing for FWHM ---
    # Extract the slice at the calculated theoretical focal depth 'b_focal_theoretical'
    focal_depth_idx = np.argmin(np.abs(z_scan_range - b_focal_theoretical))
    theoretical_beam_profile = np.abs(theoretical_field[focal_depth_idx, :]) # Envelope

    # Normalize the theoretical beam profile for FWHM calculation
    max_val = np.max(theoretical_beam_profile)
    if max_val == 0:
        print("Warning: Theoretical beam profile is all zeros. Cannot calculate FWHM.")
        return 0.0, theoretical_beam_profile # Return 0 FWHM, or NaN
    theoretical_beam_profile_norm = theoretical_beam_profile / max_val

    # Calculate FWHM using estimate_fwhm_F2 (from fwhm_methods_addon)
    fwhm_data = {
        'x_axis_mm': x_scan_range,
        'envelope_normalized': theoretical_beam_profile_norm
    }
    
    theoretical_fwhm_value = estimate_fwhm_F2(fwhm_data) ### CHANGE: Call F2

    return theoretical_fwhm_value, theoretical_beam_profile