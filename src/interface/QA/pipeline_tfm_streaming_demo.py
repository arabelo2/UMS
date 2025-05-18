#!/usr/bin/env python3
"""
pipeline_tfm_streaming_demo.py

A memory-efficient TFM back-projection pipeline:
  - caches each element’s A-scan once (O(M) simulator calls)
  - streams through Tx/Rx pairs and builds the image on the fly
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from scipy.signal import hilbert
from application.rs_2Dv_service import run_rs_2Dv_service

def compute_tof_and_cache(M, pitch, b, f, c, z_mm):
    """
    Compute common time axis and cache per-element A-scans.
    Returns:
      tof_us : (Nz,) time axis in µs
      p_cache: dict[i] → (Nz,) real A-scan for element i
    """
    # element lateral positions [mm]
    elem_pos = (np.arange(M) - (M - 1) / 2) * pitch

    # common time axis [µs] (two-way for each depth sample)
    tof_us = 2 * z_mm * 1e-3 / c * 1e6

    # cache each one-way A-scan once
    p_cache = {}
    for idx, x_elem in enumerate(elem_pos):
        # simulator returns complex pressure; we take real part & flatten
        raw = run_rs_2Dv_service(b, f, c, x_elem, 0.0, z_mm)
        p_cache[idx] = np.real(raw).squeeze()

    return tof_us, p_cache

def tfm_backprojection_streaming(M, pitch, x_im, z_im, tof_us, p_cache, c):
    """
    Perform TFM back-projection by streaming over Tx/Rx pairs.
    Returns:
      img : (Nz_im, Nx_im) normalized image
    """
    # envelope-detect each cached A-scan
    elem_pos = (np.arange(M) - (M - 1) / 2) * pitch
    Nx, Nz = x_im.size, z_im.size
    img = np.zeros((Nz, Nx), dtype=float)

    # precompute envelope of each element once
    env_cache = {
        i: np.abs(hilbert(p_cache[i], axis=-1))
        for i in p_cache
    }

    # stream through all Tx/Rx pairs
    for tx in range(M):
        for rx in range(M):
            # combine envelopes (both 1D arrays of length Nz)
            env = env_cache[tx] + env_cache[rx]
            # accumulate projection for this pair
            for ix, x in enumerate(x_im):
                # vectorize over depth to avoid inner iz‐loop
                d_tx = np.hypot(x - elem_pos[tx], z_im) * 1e-3
                d_rx = np.hypot(x - elem_pos[rx], z_im) * 1e-3
                t_us = (d_tx + d_rx) / c * 1e6
                # interpolate at once for all depths
                img[:, ix] += np.interp(t_us, tof_us, env,
                                        left=0.0, right=0.0)

    # normalize
    img /= img.max()
    return img

def main():
    # --- array / medium parameters ---
    M     = 16          # number of elements
    pitch = 0.5         # mm
    b     = pitch / 2   # half‐element width mm
    f     = 5.0         # MHz
    c     = 1500.0      # m/s

    # --- depth sampling for FMC acquisition ---
    z_min, z_max, Nz = 5.0, 60.0, 512
    z_mm = np.linspace(z_min, z_max, Nz)

    print("Caching one-way A-scans…")
    tof_us, p_cache = compute_tof_and_cache(M, pitch, b, f, c, z_mm)

    # --- imaging grid for TFM back-projection ---
    x_min, x_max, Nx = -8.0, 8.0, 200
    z_im, Nz_im       = z_min, 100
    x_im = np.linspace(x_min, x_max, Nx)
    z_im = np.linspace(z_min, z_max, Nz_im)

    print("Reconstructing TFM image (streaming)…")
    img = tfm_backprojection_streaming(
        M, pitch, x_im, z_im, tof_us, p_cache, c
    )

    # --- display ---
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 8))
    plt.imshow(img, extent=[x_min, x_max, z_max, z_min],
               aspect='auto', cmap='jet')
    plt.colorbar(label="Normalized amplitude")
    plt.xlabel("Lateral (mm)")
    plt.ylabel("Depth (mm)")
    plt.title("Streaming TFM Back‐Projection")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
