#!/usr/bin/env python3
"""
pipeline_tfm_backprojection_demo.py

Demonstration of FMC acquisition with RS_2Dv (application layer)
and TFM back‐projection reconstruction.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

from application.delay_laws2D_service import run_delay_laws2D_service
from application.rs_2Dv_service import run_rs_2Dv_service

def compute_fmc(M, pitch, b, f, c, z_mm):
    """
    Compute full‐matrix‐capture (FMC) A‐scans for a linear array.
    Returns:
      tof_us : (Nz,) time axis in µs
      FMC    : (M, M, Nz) real A‐scan data for each tx/rx pair
    """
    # element lateral positions [mm]
    elem_pos = (np.arange(M) - (M - 1) / 2) * pitch

    # common time‐axis for all A‐scans [µs]
    tof_us = 2 * z_mm * 1e-3 / c * 1e6  # 2-way travel

    # allocate FMC cube
    FMC = np.zeros((M, M, z_mm.size), dtype=float)

    for tx in range(M):
        e_tx = elem_pos[tx]
        # TX A-scan
        p_tx = run_rs_2Dv_service(b, f, c, e_tx, 0.0, z_mm)
        for rx in range(M):
            e_rx = elem_pos[rx]
            # RX A-scan
            p_rx = run_rs_2Dv_service(b, f, c, e_rx, 0.0, z_mm)
            # sum (real part) for reciprocity
            FMC[tx, rx, :] = np.real(p_tx + p_rx)

    return tof_us, FMC

def tfm_backprojection(M, pitch, x_im, z_im, tof_us, FMC, c):
    """
    Perform TFM back‐projection on the FMC data.
    Returns:
      img : (Nz_im, Nx_im) reconstructed image (normalized)
    """
    # envelope‐detect each A‐scan
    env = np.abs(hilbert(FMC, axis=-1))

    # element positions [mm]
    elem_pos = (np.arange(M) - (M - 1) / 2) * pitch

    Nx = x_im.size
    Nz = z_im.size
    img = np.zeros((Nz, Nx), dtype=float)

    # back‐project every pixel
    for ix, x in enumerate(x_im):
        for iz, z in enumerate(z_im):
            s = 0.0
            for tx in range(M):
                for rx in range(M):
                    # two-way distance in meters
                    d_tx = np.hypot(x - elem_pos[tx], z) * 1e-3
                    d_rx = np.hypot(x - elem_pos[rx], z) * 1e-3
                    # time in µs
                    t_us = (d_tx + d_rx) / c * 1e6
                    s += np.interp(t_us, tof_us, env[tx, rx, :], left=0.0, right=0.0)
            img[iz, ix] = s

    # normalize
    img /= img.max()
    return img

def main():
    # --- array / medium parameters ---
    M     = 16          # elements
    pitch = 0.5         # mm
    b     = pitch / 2   # half‐element width mm
    f     = 5.0         # MHz
    c     = 1500.0      # m/s

    # --- depth sampling for FMC acquisition ---
    z_min, z_max, Nz = 5.0, 60.0, 512
    z_mm = np.linspace(z_min, z_max, Nz)

    print("Computing FMC dataset…")
    tof_us, FMC = compute_fmc(M, pitch, b, f, c, z_mm)

    # --- imaging grid for TFM back‐projection ---
    x_min, x_max, Nx = -8.0, 8.0, 200
    z_im  , Nz_im    = 5.0 , 100
    x_im = np.linspace(x_min, x_max, Nx)
    z_im = np.linspace(z_min, z_max, Nz_im)

    print("Reconstructing TFM image…")
    img = tfm_backprojection(M, pitch, x_im, z_im, tof_us, FMC, c)

    # --- display ---
    plt.figure(figsize=(6, 8))
    plt.imshow(img, extent=[x_min, x_max, z_max, z_min],
               aspect='auto', cmap='jet')
    plt.colorbar(label="Normalized amplitude")
    plt.xlabel("Lateral (mm)")
    plt.ylabel("Depth (mm)")
    plt.title("TFM Back‐Projection Reconstruction")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
