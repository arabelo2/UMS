#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# Configuration
results_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../results/tfm_synthetic_flaw')
)
x_target_mm   = 50
flaw_depth_mm = 30

# Load lateral (x) and depth (z) grids
x_mm = pd.read_csv(os.path.join(results_dir, "x_grid.csv"))["x (mm)"].values
z_mm = pd.read_csv(os.path.join(results_dir, "z_grid.csv"))["z (mm)"].values

# Load the TFM image as a true 200×200 array
# Let pandas infer column headers, so we don't end up with an extra row of data
tfm_df    = pd.read_csv(os.path.join(results_dir, "tfm_image.csv"))
tfm_image = tfm_df.values  # shape should be (200, 200)

# Extract the 1-D RF line at the requested lateral position
ix       = np.argmin(np.abs(x_mm - x_target_mm))
rf_slice = tfm_image[:, ix]

# Compute the analytic signal envelope
env_slice = np.abs(hilbert(rf_slice))

# Sanity check: ensure z_mm and rf_slice match
if len(z_mm) != len(rf_slice):
    # truncate or pad (here we truncate)
    z_mm = z_mm[: len(rf_slice)]

# Plot
plt.figure(figsize=(10, 5))
plt.plot(z_mm, rf_slice, label="Raw TFM (linear RF)")
plt.plot(z_mm, env_slice, label="Hilbert Envelope", linewidth=2)
plt.axvline(flaw_depth_mm, color='r', linestyle='--', label="Flaw Depth")
plt.xlabel("Depth (mm)")
plt.ylabel("Amplitude (a.u.)")
plt.title(f"1-D RF vs. Envelope at x = {x_target_mm} mm")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save
out_path = os.path.join(results_dir, "rf_vs_envelope_plot.png")
plt.savefig(out_path)
plt.close()

# Debug output
print("\n=== RF vs Envelope Debug ===")
print(f"  RF slice max at depth:      {z_mm[np.argmax(rf_slice)]:.1f} mm")
print(f"  Envelope slice max at depth:{z_mm[np.argmax(env_slice)]:.1f} mm")
print(f"Saved plot to: {out_path}")
