import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert

"""
TFM Envelope Comparison Plotter
Loads the raw TFM image and its Hilbert envelope, 
then displays them side-by-side in dB.
"""

# 1) Point this to your simulation results folder:
results_subdir = "tfm_synthetic_flaw"
base_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__),
                 "../../results", results_subdir)
)

# 2) Load grids and image
x = pd.read_csv(os.path.join(base_dir, "x_grid.csv"))["x (mm)"].values
z = pd.read_csv(os.path.join(base_dir, "z_grid.csv"))["z (mm)"].values
tfm_image = pd.read_csv(os.path.join(base_dir, "tfm_image.csv"), header=0).values

# 3) Compute Hilbert envelope along depth (axis=0)
analytic = hilbert(tfm_image, axis=0)
env = np.abs(analytic)

# 4) Convert both to dB
tfm_db = 20 * np.log10(np.abs(tfm_image) + 1e-6)
env_db = 20 * np.log10(env + 1e-6)

# 5) Plot side-by-side
X, Z = np.meshgrid(x, z)
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

im0 = ax0.imshow(tfm_db, extent=[x.min(), x.max(), z.max(), z.min()],
                 cmap="hot", vmin=-40, vmax=0, aspect="auto")
ax0.set_title("Original TFM Image (dB)")
ax0.set_xlabel("Lateral (mm)")
ax0.set_ylabel("Depth (mm)")
fig.colorbar(im0, ax=ax0, label="Amplitude (dB)")

im1 = ax1.imshow(env_db, extent=[x.min(), x.max(), z.max(), z.min()],
                 cmap="hot", vmin=-40, vmax=0, aspect="auto")
ax1.set_title("TFM with Hilbert Envelope (dB)")
ax1.set_xlabel("Lateral (mm)")
fig.colorbar(im1, ax=ax1, label="Amplitude (dB)")

# 6) Mark the known flaw position
for ax in (ax0, ax1):
    ax.scatter([50], [30], c="cyan", marker="x", label="Flaw @ (50,30) mm")
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()
