#!/usr/bin/env python3
"""
pipeline_scatterer_fwhm_demo_optimized.py

1) Cache each element’s A‐scan once (memoization).
2) Stream raw TFM back‐projection + inject single‐sample scatterer.
3) Compute 2D envelope of the final raw image.
4) Measure lateral & axial FWHM on the smooth envelope.
5) Plot raw vs. envelope side by side.
"""
import sys, os
import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt

# application layer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from application.rs_2Dv_service import run_rs_2Dv_service

def compute_tof_and_cache(M, pitch, b, f, c, z_mm):
    """Compute common tof_us and cache real A‐scans per element."""
    elem_pos = (np.arange(M) - (M-1)/2) * pitch
    tof_us   = 2 * z_mm * 1e-3 / c * 1e6

    p_cache = {}
    for i, x_elem in enumerate(elem_pos):
        raw = run_rs_2Dv_service(b, f, c, x_elem, 0.0, z_mm)
        p_cache[i] = np.real(raw).squeeze()
    return tof_us, p_cache

def raw_tfm_streaming_scatterer(M, pitch, x_im, z_im,
                                tof_us, p_cache, c,
                                scatterer_z, scatterer_amp):
    """
    Stream raw TFM back‐projection, inject
    scatterer tick into the raw A‐scan sum for each pair.
    """
    elem_pos = (np.arange(M) - (M-1)/2) * pitch
    Nx, Nz   = x_im.size, z_im.size
    img      = np.zeros((Nz, Nx), dtype=float)

    # precompute nothing else—just raw sums on the fly
    for tx in range(M):
        for rx in range(M):
            # memoized raw A‐scan sum for this pair
            raw_sum = p_cache[tx] + p_cache[rx]

            # inject scatterer into raw_sum
            d_tx_sc = np.hypot(0.0 - elem_pos[tx], scatterer_z) * 1e-3
            d_rx_sc = np.hypot(0.0 - elem_pos[rx], scatterer_z) * 1e-3
            t_sc_us = (d_tx_sc + d_rx_sc)/c * 1e6
            idx = np.argmin(np.abs(tof_us - t_sc_us))
            raw_sum[idx] += scatterer_amp

            # back‐project onto image grid
            for ix, x in enumerate(x_im):
                d_tx = np.hypot(x - elem_pos[tx], z_im)*1e-3
                d_rx = np.hypot(x - elem_pos[rx], z_im)*1e-3
                t_us = (d_tx + d_rx)/c * 1e6
                img[:, ix] += np.interp(t_us, tof_us, raw_sum,
                                        left=0.0, right=0.0)

    # normalize raw image
    return img / np.max(np.abs(img))

def measure_fwhm(x, profile):
    """Full‐width at half max of 1D profile."""
    half = np.max(profile) / 2.0
    inds = np.where(profile >= half)[0]
    return (x[inds[-1]] - x[inds[0]]) if inds.size >= 2 else 0.0

def main():
    # parameters
    M, pitch = 16, 0.5  # elements, mm
    b = pitch/2; f = 5.0; c = 1500.0

    # sampling grids
    z_fmc = np.linspace(5, 60, 1024)
    x_im  = np.linspace(-8, 8, 201)
    z_im  = np.linspace(5, 60, 271)

    # scatterer specs
    scatterer_z, scatterer_amp = 40.0, 1.0

    # 1) memoize A‐scans
    tof_us, p_cache = compute_tof_and_cache(M, pitch, b, f, c, z_fmc)

    # 2) raw back‐projection + scatterer injection
    raw_img = raw_tfm_streaming_scatterer(
        M, pitch, x_im, z_im,
        tof_us, p_cache, c,
        scatterer_z, scatterer_amp
    )

    # 3) envelope of final raw image (along depth axis)
    env_img = np.abs(hilbert(raw_img, axis=0))
    env_img /= env_img.max()

    # 4) measure FWHM on envelope
    iz0 = np.argmin(np.abs(z_im - scatterer_z))
    fwhm_x = measure_fwhm(x_im, env_img[iz0, :])
    ix0 = np.argmin(np.abs(x_im - 0.0))
    fwhm_z = measure_fwhm(z_im, env_img[:, ix0])

    print(f"Envelope‐based lateral FWHM: {fwhm_x:.2f} mm")
    print(f"Envelope‐based axial   FWHM: {fwhm_z:.2f} mm")

    # 5) plot side‐by‐side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    im1 = ax1.imshow(raw_img,
                     extent=[x_im[0], x_im[-1], z_im[-1], z_im[0]],
                     aspect='auto', cmap='seismic', vmin=-1, vmax=1)
    ax1.set_title("Raw TFM Reconstruction")
    ax1.set_xlabel("Lateral (mm)")
    ax1.set_ylabel("Depth (mm)")
    plt.colorbar(im1, ax=ax1, label="Normalized amplitude")

    im2 = ax2.imshow(env_img,
                     extent=[x_im[0], x_im[-1], z_im[-1], z_im[0]],
                     aspect='auto', cmap='jet', vmin=0, vmax=1)
    ax2.set_title("Envelope of Raw TFM Reconstruction")
    ax2.set_xlabel("Lateral (mm)")
    ax2.set_ylabel("Depth (mm)")
    ax2.scatter(0, scatterer_z, c='w', marker='+')
    plt.colorbar(im2, ax=ax2, label="Normalized envelope")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
