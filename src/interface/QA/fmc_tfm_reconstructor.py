# interface/fmc_tfm_reconstructor.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import scipy.io
import scipy.signal
import matplotlib.pyplot as plt

# --- Configuration from Metadata (used as fallbacks if not in .mat file) ---
FMC_FILE_PATH = 'C:/Users/HP/Downloads/FMC_2012_06_12_at_13_09.mat' # <-- Make sure this path is correct
# Set this to True to use the envelope of A-scans (recommended for clearer images)
# Set this to False to use raw RF A-scans (will sum complex values, then take abs)
USE_ENVELOPE = True

# Metadata values (provide sensible defaults or use as fallbacks)
META_SAMPLING_RATE_HZ = 100e6  # 100 MHz
META_NUM_SAMPLE_POINTS = 10000
META_ARRAY_PITCH_MM = 0.7     # mm
META_WAVE_SPEED_MPS = 5820    # m/s
META_NUM_ELEMENTS = 128

# Flaw location from metadata (in mm)
FLAW_X_MM = 35.0
FLAW_Z_MM = 20.0

# --- 1. Load Data ---
print(f"Loading FMC data from: {FMC_FILE_PATH}")
try:
    mat_data = scipy.io.loadmat(FMC_FILE_PATH)
except FileNotFoundError:
    print(f"ERROR: FMC data file not found at '{FMC_FILE_PATH}'.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading .mat file: {e}")
    sys.exit(1)

# --- Extract or Define Parameters ---
try:
    # FMC data: metadata says (tx_idx, rx_idx, time_idx)
    # We need to transpose to (time_idx, rx_idx, tx_idx) for the TFM loop
    fmc_data_raw_original_dims = mat_data['FMC_new']
    print(f"Original fmc_data dimensions: {fmc_data_raw_original_dims.shape}")
    if fmc_data_raw_original_dims.shape == (META_NUM_ELEMENTS, META_NUM_ELEMENTS, META_NUM_SAMPLE_POINTS):
        # This matches (tx, rx, time)
        fmc_data_raw = np.transpose(fmc_data_raw_original_dims, (2, 1, 0)) # (time, rx, tx)
        print(f"Transposed fmc_data to: {fmc_data_raw.shape}")
    elif fmc_data_raw_original_dims.shape == (META_NUM_SAMPLE_POINTS, META_NUM_ELEMENTS, META_NUM_ELEMENTS):
        # This already matches (time, rx, tx) or (time, tx, rx)
        # Assuming the latter dimensions in the loop are (rx, tx)
        fmc_data_raw = fmc_data_raw_original_dims
        print(f"fmc_data dimensions already (time, el, el): {fmc_data_raw.shape}")
    else:
        print(f"ERROR: Unexpected fmc_data dimensions: {fmc_data_raw_original_dims.shape}")
        print(f"Expected ({META_NUM_ELEMENTS}, {META_NUM_ELEMENTS}, {META_NUM_SAMPLE_POINTS}) or ({META_NUM_SAMPLE_POINTS}, {META_NUM_ELEMENTS}, {META_NUM_ELEMENTS}) based on metadata.")
        sys.exit(1)

    num_time_samples = fmc_data_raw.shape[0]
    num_elements = fmc_data_raw.shape[1] # Should be number of receivers
    if fmc_data_raw.shape[2] != num_elements:
        print(f"ERROR: Number of receivers ({num_elements}) does not match number of transmitters ({fmc_data_raw.shape[2]}) after transpose.")
        sys.exit(1)
    if num_elements != META_NUM_ELEMENTS:
        print(f"WARNING: Number of elements from data ({num_elements}) differs from metadata ({META_NUM_ELEMENTS}). Using data value.")

except KeyError as e:
    print(f"Error: Key 'fmc_data' not found in the .mat file.")
    sys.exit(1)


try:
    time_vector = mat_data['time_vector'].squeeze()
    if len(time_vector) != num_time_samples:
        print(f"WARNING: Length of 'time_vector' ({len(time_vector)}) from .mat file differs from fmc_data samples ({num_time_samples}). Regenerating.")
        raise KeyError # Force regeneration
except KeyError:
    print("INFO: 'time_vector' not found or inconsistent in .mat file. Generating from metadata/sampling rate.")
    # Try to get sampling rate from .mat, else use metadata
    try:
        sampling_rate_hz = float(mat_data['sampling_rate']) # Assuming key 'sampling_rate' and value in Hz
        if sampling_rate_hz != META_SAMPLING_RATE_HZ:
            print(f"INFO: Using sampling_rate from .mat: {sampling_rate_hz/1e6} MHz")
    except KeyError:
        print(f"INFO: 'sampling_rate' not found in .mat. Using metadata value: {META_SAMPLING_RATE_HZ/1e6} MHz")
        sampling_rate_hz = META_SAMPLING_RATE_HZ
    dt = 1.0 / sampling_rate_hz
    time_vector = np.arange(num_time_samples) * dt

try:
    array_pitch_m = float(mat_data['array_pitch']) # Assume in meters if from .mat
    # If you know it's in mm in the .mat file, convert: array_pitch_m /= 1000.0
    print(f"INFO: Using array_pitch from .mat file: {array_pitch_m*1000:.2f} mm")
except KeyError:
    print(f"INFO: 'array_pitch' not found in .mat file. Using metadata value: {META_ARRAY_PITCH_MM} mm.")
    array_pitch_m = META_ARRAY_PITCH_MM / 1000.0 # Convert mm to m

try:
    wave_speed_mps = float(mat_data['wave_speed']) # Assume in m/s
    print(f"INFO: Using wave_speed from .mat file: {wave_speed_mps} m/s")
except KeyError:
    print(f"INFO: 'wave_speed' not found in .mat file. Using metadata value: {META_WAVE_SPEED_MPS} m/s.")
    wave_speed_mps = META_WAVE_SPEED_MPS


# --- Prepare data based on whether to use envelope or raw RF ---
if USE_ENVELOPE:
    print("Processing: Using envelope of A-scans.")
    fmc_to_use = np.abs(scipy.signal.hilbert(fmc_data_raw, axis=0))
    dtype_for_tfm_image = np.float64
    dtype_for_pixel_sum = np.float64
else:
    print("Processing: Using raw RF A-scans.")
    fmc_to_use = fmc_data_raw
    dtype_for_tfm_image = np.complex128
    dtype_for_pixel_sum = np.complex128

# --- 2. Define Imaging Region based on Flaw Location and Array Size ---
# Metadata: 1st element at (0,0,0). Flaw at x=35mm, z=20mm.
array_total_width_m = (num_elements - 1) * array_pitch_m

# X-axis imaging window (in meters)
# Start slightly before the array if desired, or at 0. View past the flaw.
x_min_m = -0.005  # Start imaging slightly to the left of the first element
x_max_m = max(FLAW_X_MM / 1000.0 + 0.030, array_total_width_m + 0.010) # View 30mm past flaw_x or 10mm past array end

# Z-axis imaging window (depth in meters)
z_min_m = 0.001  # Start just below the surface (0.001 m = 1mm)
z_max_m = FLAW_Z_MM / 1000.0 + 0.030  # View 30mm deeper than the flaw_z (e.g., up to 50mm depth if flaw at 20mm)

num_x_pixels = 250 # Adjust for desired resolution
num_z_pixels = 300 # Adjust for desired resolution

x_coords = np.linspace(x_min_m, x_max_m, num_x_pixels)
z_coords = np.linspace(z_min_m, z_max_m, num_z_pixels)

print(f"Imaging Region: X from {x_min_m*1000:.1f}mm to {x_max_m*1000:.1f}mm ({num_x_pixels} pixels)")
print(f"Imaging Region: Z from {z_min_m*1000:.1f}mm to {z_max_m*1000:.1f}mm ({num_z_pixels} pixels)")

tfm_image = np.zeros((num_z_pixels, num_x_pixels), dtype=dtype_for_tfm_image)

# --- 3. Calculate Element Positions ---
# Metadata: 1st element at (0,0,0)
element_x_pos = np.arange(num_elements) * array_pitch_m
element_z_pos = np.zeros(num_elements) # Elements on the surface z=0

# --- 4. TFM Calculation ---
print("Starting TFM calculation...")
for iz, zp in enumerate(z_coords):
    if iz % 20 == 0 and iz > 0:
        print(f"Processing depth layer {iz}/{num_z_pixels-1}...")
    for ix, xp in enumerate(x_coords):
        pixel_sum = np.array(0.0, dtype=dtype_for_pixel_sum)
        for t_idx in range(num_elements):
            xt = element_x_pos[t_idx]
            # zt = element_z_pos[t_idx] # This is 0

            dist_tx_pixel = np.sqrt((xp - xt)**2 + (zp - 0.0)**2) # zp - element_z_pos[t_idx]
            tof_tx_pixel = dist_tx_pixel / wave_speed_mps

            for r_idx in range(num_elements):
                xr = element_x_pos[r_idx]
                # zr = element_z_pos[r_idx] # This is 0

                dist_pixel_rx = np.sqrt((xp - xr)**2 + (zp - 0.0)**2) # zp - element_z_pos[r_idx]
                tof_pixel_rx = dist_pixel_rx / wave_speed_mps

                total_tof = tof_tx_pixel + tof_pixel_rx
                ascan_tr = fmc_to_use[:, r_idx, t_idx]
                interpolated_amplitude = np.interp(total_tof, time_vector, ascan_tr, left=0.0, right=0.0)
                pixel_sum += interpolated_amplitude
        tfm_image[iz, ix] = pixel_sum
print("TFM calculation complete.")

# --- Final Image Preparation ---
if not USE_ENVELOPE:
    final_tfm_image = np.abs(tfm_image)
else:
    final_tfm_image = tfm_image

# --- 5. Display Image ---
print("Displaying image...")
plt.figure(figsize=(10, 8))
plt.imshow(final_tfm_image, aspect='auto',
           extent=[x_min_m * 1000, x_max_m * 1000, z_max_m * 1000, z_min_m * 1000]) # Display in mm
plt.xlabel("X position (mm)")
plt.ylabel("Z position (mm)")
title_str = "TFM Reconstructed Image"
if USE_ENVELOPE: title_str += " (Envelope)"
else: title_str += " (Raw RF, abs)"
# Add flaw location marker
plt.scatter(FLAW_X_MM, FLAW_Z_MM, s=100, c='red', marker='x', label=f'Flaw Tip ({FLAW_X_MM}mm, {FLAW_Z_MM}mm)')
plt.legend()
plt.title(title_str)
plt.colorbar(label="Intensity")
plt.show()

print("Script finished.")