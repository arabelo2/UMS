# UMS/src/interface/utils.py
import numpy as np
from scipy.signal import hilbert

def calculate_envelope_fwhm(signal_data, z_vals_data):
    """
    Calculates the envelope and Full Width at Half Maximum (FWHM) of a signal.

    Args:
        signal_data (np.ndarray): The input signal (can be complex or real).
                                  If complex, its absolute value is used.
        z_vals_data (np.ndarray): The corresponding z-values (e.g., depth).

    Returns:
        tuple: (np.ndarray, float)
            - envelope (np.ndarray): The calculated envelope of the signal.
            - fwhm (float): The calculated FWHM in the units of z_vals.
                            Returns 0.0 if FWHM cannot be determined.
    """
    if len(signal_data) != len(z_vals_data) or len(signal_data) == 0:
        # print("Warning: Signal and z_vals must have the same non-zero length for FWHM calculation.")
        return np.array([]), 0.0

    # Ensure signal_data and z_vals_data are 1D numpy arrays
    signal = np.array(signal_data).ravel()
    z_vals = np.array(z_vals_data).ravel()

    signal_mag = np.abs(signal) if np.iscomplexobj(signal) else signal
    
    # Avoid issues with all-zero signals for hilbert transform
    if np.all(signal_mag < 1e-9): # Check if all values are effectively zero
        # print("Warning: Signal is all zeros or near-zeros. Cannot calculate FWHM.")
        return signal_mag, 0.0

    try:
        analytic_signal = hilbert(signal_mag)
        envelope = np.abs(analytic_signal)
    except ValueError as e: # Catch hilbert transform errors (e.g. for very short signals)
        # print(f"Warning: Hilbert transform error: {e}. Using magnitude as envelope.")
        envelope = signal_mag # Fallback to magnitude

    if len(envelope) == 0:
        return np.array([]), 0.0
        
    peak_idx = np.argmax(envelope)
    peak_val = envelope[peak_idx]

    if peak_val < 1e-9: # If peak is effectively zero
        # print("Warning: Peak of envelope is zero or near-zero. FWHM is 0.")
        return envelope, 0.0

    half_max = peak_val / 2.0
    
    # Find indices where envelope crosses half_max
    above_half_max = envelope >= half_max  # Use >= to catch points exactly at half_max
    
    # Find all crossings (both up and down)
    crossings_indices = np.where(np.diff(above_half_max))[0]

    if len(crossings_indices) < 2:
        # Not enough crossings to define FWHM (e.g., signal never drops below half_max after rising)
        # print("Warning: Not enough crossings to determine FWHM.")
        return envelope, 0.0

    # We are interested in the first rise above half_max and the last fall below it around the main peak
    # This can be tricky if there are multiple peaks above half_max.
    # A common approach is to find the crossings closest to the main peak.

    # Find first crossing going up before or at peak_idx
    left_crossings = crossings_indices[crossings_indices < peak_idx]
    idx1_left = left_crossings[-1] if len(left_crossings) > 0 else -1
    
    # Find first crossing going down after or at peak_idx
    right_crossings = crossings_indices[crossings_indices >= peak_idx]
    idx1_right = right_crossings[0] if len(right_crossings) > 0 else -1

    z_left, z_right = None, None

    # Interpolate left FWHM point
    if idx1_left != -1 and idx1_left + 1 < len(z_vals):
        x_pts = z_vals[idx1_left : idx1_left+2]
        y_pts = envelope[idx1_left : idx1_left+2]
        if y_pts[1] != y_pts[0]: # Avoid division by zero if flat
            z_left = np.interp(half_max, [y_pts[0], y_pts[1]], [x_pts[0], x_pts[1]])
        elif y_pts[0] >= half_max: # If flat at or above half_max
             z_left = x_pts[0]


    # Interpolate right FWHM point
    if idx1_right != -1 and idx1_right + 1 < len(z_vals):
        x_pts = z_vals[idx1_right : idx1_right+2]
        y_pts = envelope[idx1_right : idx1_right+2]
        if y_pts[1] != y_pts[0]: # Avoid division by zero
            # Order for interpolation: if y_pts[0] > y_pts[1] (going down)
            z_right = np.interp(half_max, [y_pts[1], y_pts[0]], [x_pts[1], x_pts[0]])
        elif y_pts[0] >= half_max: # If flat at or above half_max
            z_right = x_pts[1]


    if z_left is not None and z_right is not None and z_right > z_left:
        fwhm = z_right - z_left
        return envelope, fwhm
    else:
        # Fallback if interpolation failed or order is wrong
        # Try a simpler method of finding first and last point above half_max
        indices_above = np.where(above_half_max)[0]
        if len(indices_above) > 0:
            z_min_above = z_vals[indices_above[0]]
            z_max_above = z_vals[indices_above[-1]]
            # This is not strictly FWHM but width of region above half_max
            # A more robust FWHM might require peak fitting for complex envelopes
            # For now, if interpolation failed, this is a rough estimate or return 0
            # print(f"Warning: FWHM interpolation was inconclusive (z_left={z_left}, z_right={z_right}). Using width above half max or 0.")
            # return envelope, max(0, z_max_above - z_min_above) # This is an alternative but less accurate
            return envelope, 0.0 # Default to 0 if robust interpolation fails
        else:
            return envelope, 0.0