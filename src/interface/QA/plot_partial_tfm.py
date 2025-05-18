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
tx_idx = 16
x_target_mm = 50
flaw_depth_mm = 30

# Load grids and image
x_mm = pd.read_csv(os.path.join(results_dir, "x_grid.csv"))["x (mm)"].values
z_mm = pd.read_csv(os.path.join(results_dir, "z_grid.csv"))["z (mm)"].values

# Read the TFM image (skip the header row so data dims remain 200×200)
tfm_df    = pd.read_csv(os.path.join(results_dir, "tfm_image.csv"))
tfm_image = tfm_df.values  # shape: (200, 200)

# Extract partial TFM at the target lateral position
ix = np.argmin(np.abs(x_mm - x_target_mm))
partial = tfm_image[:, ix]          # slice over depth
env     = np.abs(hilbert(partial))  # Hilbert envelope

# Plot
plt.figure(figsize=(10, 5))
plt.plot(z_mm, partial, label=f"Partial TFM Slice at x={x_target_mm} mm")
plt.plot(z_mm, env,     label="Hilbert Envelope", linewidth=2)
plt.axvline(flaw_depth_mm, color='r', linestyle='--', label='Flaw Depth')
plt.xlabel("Depth (mm)")
plt.ylabel("Amplitude (a.u.)")
plt.title(f"Partial TFM Slice at x={x_target_mm} mm (Tx index {tx_idx})")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the figure
output_path = os.path.join(results_dir, f'partial_tfm_plot_tx{tx_idx}.png')
plt.savefig(output_path)
plt.close()

print(f"Saved partial TFM plot to {output_path}")
