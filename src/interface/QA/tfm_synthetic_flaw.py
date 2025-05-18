#!/usr/bin/env python3
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Parameters ──
num_elements = 32
num_samples = 2048  # Increased for better time resolution
sampling_rate = 40e6  # Hz
center_freq_mhz = 5  # MHz
flaw_depth_mm = 30  # mm
flaw_x_mm = 50  # mm
overlay_vel = 5900  # m/s (steel)

# Path setup
script_name = os.path.splitext(os.path.basename(__file__))[0]
save_dir    = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../results', script_name)
)
os.makedirs(save_dir, exist_ok=True)

# Time vector
time_us = np.arange(num_samples) / sampling_rate * 1e6

# ── Generate Synthetic FMC Data ──
def generate_echo(t0_us, freq_mhz, amp, dur_us):
    """Create Gaussian-windowed sinusoidal echo"""
    envelope = np.exp(-((time_us - t0_us)**2) / (2 * dur_us**2))
    wave = np.sin(2 * np.pi * freq_mhz * (time_us - t0_us))
    return amp * envelope * wave

# Calculate expected flaw echo timing
round_trip_us = 2 * (flaw_depth_mm/1000) / overlay_vel * 1e6
fmc = 0.005 * np.random.randn(num_elements, num_elements, num_samples)

# Inject flaw echo
duration_us = 0.3
for tx in range(num_elements):
    for rx in range(num_elements):
        fmc[tx, rx] += generate_echo(round_trip_us, center_freq_mhz, 1.0, duration_us)

# Save FMC data
for tx in range(num_elements):
    for rx in range(num_elements):
        pd.DataFrame({
            'Time (us)': time_us,
            'Signal': fmc[tx, rx]
        }).to_csv(os.path.join(save_dir, f'fmc_tx{tx}_rx{rx}.csv'), index=False)

# ── Imaging Grid & Array Geometry ──
x_mm = np.linspace(0, 100, 200)  # Lateral positions
z_mm = np.linspace(0, 60, 200)   # Depth positions
pitch_mm = 0.6
center_x = 50

elem_idx = np.arange(num_elements) - (num_elements-1)/2
elem_pos_x = center_x + elem_idx * pitch_mm

# Save grids
pd.DataFrame({'x (mm)': x_mm}).to_csv(os.path.join(save_dir,'x_grid.csv'), index=False)
pd.DataFrame({'z (mm)': z_mm}).to_csv(os.path.join(save_dir,'z_grid.csv'), index=False)

# ── TFM Reconstruction ──
def interp_frac(sig, tof_us, tvec_us):
    """Safe interpolation with bounds checking"""
    return np.interp(tof_us, tvec_us, sig, left=0.0, right=0.0)

Xg, Zg = x_mm.reshape(1,-1), z_mm.reshape(-1,1)  # Grid broadcasting
tfm_image = np.zeros((len(z_mm), len(x_mm)))

# Debug prints
print("\n=== TFM Reconstruction Debug ===")
print(f"Expected flaw TOF: {round_trip_us:.2f} μs")
print(f"Time vector range: {time_us[0]:.2f}-{time_us[-1]:.2f} μs")
print(f"Flaw position: x={flaw_x_mm}mm, z={flaw_depth_mm}mm")

# Load and process FMC data
for tx in range(num_elements):
    xi_mm = elem_pos_x[tx]
    for rx in range(num_elements):
        xj_mm = elem_pos_x[rx]
        sig = pd.read_csv(os.path.join(save_dir, f'fmc_tx{tx}_rx{rx}.csv'))['Signal'].values
        
        # Calculate TOF map
        d_tx = np.sqrt((Xg - xi_mm)**2 + Zg**2)
        d_rx = np.sqrt((Xg - xj_mm)**2 + Zg**2)
        tof_us = (d_tx + d_rx)/1000/overlay_vel*1e6
        
        # Debug sample point
        if tx == rx == num_elements//2:
            flaw_idx = np.argmin(np.abs(z_mm - flaw_depth_mm))
            print(f"Central element TOF at flaw: {tof_us[flaw_idx, np.argmin(np.abs(x_mm - flaw_x_mm))]:.2f} μs")
        
        tfm_image += interp_frac(sig, tof_us, time_us)

# Normalize and save
tfm_image /= num_elements**2
pd.DataFrame(tfm_image).to_csv(os.path.join(save_dir,'tfm_image.csv'), index=False)

# Final debug output
flaw_z_idx = np.argmin(np.abs(z_mm - flaw_depth_mm))
flaw_x_idx = np.argmin(np.abs(x_mm - flaw_x_mm))
print(f"\nTFM image stats:")
depth_idx, lateral_idx = np.unravel_index(np.argmax(tfm_image), tfm_image.shape)
print(f"Max value: {tfm_image[depth_idx,lateral_idx]:.2f} at depth {z_mm[depth_idx]:.1f} mm, lateral {x_mm[lateral_idx]:.1f} mm")
print(f"Value at flaw: {tfm_image[flaw_z_idx, flaw_x_idx]:.2f}")

# Create debug plot
plt.figure(figsize=(10,5))
plt.imshow(tfm_image, extent=[x_mm.min(), x_mm.max(), z_mm.max(), z_mm.min()], 
           aspect='equal', cmap='viridis')
plt.colorbar(label='Amplitude')
plt.scatter([flaw_x_mm], [flaw_depth_mm], c='red', marker='x', s=100, label='Flaw')
plt.title('TFM Image with Flaw Marker')
plt.xlabel('Lateral Position (mm)')
plt.ylabel('Depth (mm)')
plt.legend()
plt.savefig(os.path.join(save_dir, 'tfm_debug_plot.png'))
plt.close()