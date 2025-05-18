#!/usr/bin/env python3
"""
pipeline_scatterer_demo.py

Builds a synthetic FMC dataset with a single point scatterer, sums it into an A-scan
for sanity checking, and then forms a TFM image.  Drop in your own TFM back‑projection
function where indicated.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# make sure we can import from your project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from application.delay_laws2D_service import run_delay_laws2D_service
from application.rs_2Dv_service      import run_rs_2Dv_service
# from application.tfm_service        import run_tfm_backprojection  # <-- your TFM routine

def compute_fmc_with_scatterer(M, pitch, b, f, c, z_mm,
                               scatterer_z=30.0, scatterer_amp=1.0):
    """
    Build an FMC cube with a single point‑scatterer at depth `scatterer_z` (mm).
    Returns:
      tof_us     : 1D time axis (µs)
      summed_rf  : summed A-scan RF trace
      summed_env : envelope of summed_rf
      FMC        : full FMC array (M×M×len(z_mm))
    """
    # 1) per-element TX delays (µs) (no steering, no focus)
    tof_tx = run_delay_laws2D_service(M, pitch, 0.0, np.inf, c)  # shape (M,)

    # 2) time axis for depth samples (µs)
    tof_us = 2.0 * z_mm * 1e-3 / c * 1e6

    # 3) allocate FMC: TX × RX × time
    FMC = np.zeros((M, M, tof_us.size), dtype=float)

    for tx in range(M):
        for rx in range(M):
            # total receive delay (mirror TX delays)
            rx_delay = tof_tx[rx]

            # lateral offset of TX element
            e = (tx - (M - 1) / 2) * pitch

            # raw A-scan from Rayleigh-Sommerfeld 2Dv
            p = run_rs_2Dv_service(b, f, c, e, 0.0, z_mm)  # complex output
            # squeeze any singleton dimensions to 1D
            p = np.squeeze(p)

            # build a delta at the scatterer depth
            scatter = np.zeros_like(p.real)
            # find nearest depth index
            idx = np.argmin(np.abs(z_mm - scatterer_z))
            scatter[idx] = scatterer_amp

            # echo = element response × scatterer
            echo = p.real * scatter

            # align and insert into FMC cube
            xp = tof_us + tof_tx[tx] + rx_delay
            FMC[tx, rx, :] = np.interp(tof_us, xp, echo,
                                        left=0.0, right=0.0)

    # sum over all TX–RX to get one A-scan
    summed_rf  = FMC.sum(axis=(0, 1))
    summed_env = np.abs(hilbert(summed_rf))

    return tof_us, summed_rf, summed_env, FMC


def plot_ascan(tof_us, rf, env):
    """Plot RF and normalized envelope."""
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8,5))
    ax1.plot(tof_us, rf, 'b-')
    ax1.set_ylabel("Summed RF", fontsize=12)
    ax1.grid(True)

    ax2.plot(tof_us, env/np.max(env), 'r-')
    ax2.set_xlabel("Time (µs)", fontsize=12)
    ax2.set_ylabel("Normalized Envelope", fontsize=12)
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def plot_tfm_image(x_grid, z_grid, image):
    """Simple imshow of a TFM image (lateral x vs. depth z)."""
    plt.figure(figsize=(6,8))
    plt.imshow(image, extent=[x_grid.min(), x_grid.max(),
                               z_grid.max(), z_grid.min()],
               cmap='jet', aspect='auto')
    plt.xlabel("Lateral (mm)")
    plt.ylabel("Depth (mm)")
    plt.title("TFM Reconstruction")
    plt.colorbar(label="Amplitude")
    plt.show()


def main():
    # --- parameters ---
    M       = 16          # number of elements
    pitch   = 0.5         # element pitch (mm)
    b       = pitch / 2   # half-element width (mm)
    f       = 5.0         # frequency (MHz)
    c       = 1500.0      # sound speed (m/s)
    z_mm    = np.linspace(5, 60, 512)   # depth axis (mm)

    # scatterer settings
    scatterer_z   = 30.0  # mm
    scatterer_amp = 1.0

    # compute FMC and summed A-scan
    tof_us, rf, env, FMC = compute_fmc_with_scatterer(
        M, pitch, b, f, c, z_mm, scatterer_z, scatterer_amp
    )

    # plot the summed A-scan
    plot_ascan(tof_us, rf, env)

    # build a stub TFM image (replace with your backprojection)
    x_grid = np.linspace(-8, 8, 256)
    z_grid = z_mm
    img    = np.zeros((z_grid.size, x_grid.size))

    plot_tfm_image(x_grid, z_grid, img)

if __name__ == "__main__":
    main()
