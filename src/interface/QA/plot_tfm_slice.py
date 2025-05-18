import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert

"""
TFM 1-D Slice Comparison Plotter
Loads the raw TFM image and its Hilbert envelope,
extracts a slice at a given lateral position,
and plots both RF and envelope profiles versus depth.
"""

# ——— Configuration ———
results_subdir = "tfm_synthetic_flaw"   # name of your sim folder under results/
results_dir = os.path.abspath(os.path.join("results", results_subdir))

# ——— Load Grids & Image ———
x = pd.read_csv(os.path.join(results_dir, "x_grid.csv"))["x (mm)"].values
z = pd.read_csv(os.path.join(results_dir, "z_grid.csv"))["z (mm)"].values
tfm_image = pd.read_csv(os.path.join(results_dir, "tfm_image.csv"), header=0).values

# ——— Compute Envelope ———
analytic = hilbert(tfm_image, axis=0)
env = np.abs(analytic)

# ——— Convert to dB ———
tfm_db   = 20 * np.log10(np.abs(tfm_image) + 1e-6)
env_db   = 20 * np.log10(env + 1e-6)

# ——— Extract 1-D Slice ———
target_x = 50  # mm
ix = np.argmin(np.abs(x - target_x))
slice_tfm = tfm_db[:, ix]
slice_env = env_db[:, ix]

# ——— Plot ———
plt.figure(figsize=(6,5))
plt.plot(z, slice_tfm, label="Raw TFM (dB)")
plt.plot(z, slice_env, label="Hilbert Envelope (dB)")
plt.xlabel("Depth (mm)")
plt.ylabel("Amplitude (dB)")
plt.title(f"1-D TFM Slice at x = {target_x} mm")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
