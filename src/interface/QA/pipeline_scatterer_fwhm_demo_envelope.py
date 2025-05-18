#!/usr/bin/env python3
"""
pipeline_scatterer_fwhm_demo_envelope.py

Same as the original scatterer demo, but after raw TFM back‐projection:
  - take the 2D Hilbert envelope of the final image
  - measure lateral & axial FWHM on that envelope
  - plot raw vs. envelope for comparison
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# ensure application layer on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from application.delay_laws2D_service import run_delay_laws2D_service
from application.rs_2Dv_service      import run_rs_2Dv_service

def compute_fmc_with_scatterer(M, pitch, b, f, c, z_mm, scatterer_z, scatterer_amp):
    """Identical to your original FMC+catterer injector."""
    tof_us = 2 * z_mm * 1e-3 / c * 1e6
    N = len(z_mm)
    FMC = np.zeros((M, M, N), dtype=float)

    # fill with monostatic RS_2Dv responses
    for tx in range(M):
        _ = run_delay_laws2D_service(M, pitch, 0.0, np.inf, c)  # unused
        for rx in range(M):
            raw = run_rs_2Dv_service(b, f, c, 0.0, 0.0, z_mm)
            FMC[tx, rx, :] = np.real(raw)

    # inject point scatterer
    x_sc = 0.0
    elem_pos = (np.arange(M) - (M-1)/2) * pitch
    for tx in range(M):
        x_tx = elem_pos[tx]
        for rx in range(M):
            x_rx = elem_pos[rx]
            d_tx = np.hypot(x_sc - x_tx, scatterer_z)
            d_rx = np.hypot(x_sc - x_rx, scatterer_z)
            t_sc = (d_tx + d_rx)*1e-3/c*1e6
            idx  = np.argmin(np.abs(tof_us - t_sc))
            FMC[tx, rx, idx] += scatterer_amp

    return tof_us, FMC

def tfm_backprojection(M, pitch, x_im, z_im, tof_us, FMC, c):
    """Identical to your original raw TFM back‐projection."""
    Nx, Nz = len(x_im), len(z_im)
    img    = np.zeros((Nz, Nx), dtype=float)
    elem_pos = (np.arange(M) - (M-1)/2) * pitch

    for ix, xi in enumerate(x_im):
        for iz, zj in enumerate(z_im):
            s = 0.0
            for tx in range(M):
                d_tx = np.hypot(xi - elem_pos[tx], zj)*1e-3
                for rx in range(M):
                    d_rx = np.hypot(xi - elem_pos[rx], zj)*1e-3
                    t_us = (d_tx + d_rx)/c*1e6
                    s   += np.interp(t_us, tof_us, FMC[tx, rx, :],
                                     left=0.0, right=0.0)
            img[iz, ix] = s

    return img

def measure_fwhm(x, profile):
    """Same as before."""
    half = np.max(profile)/2.0
    inds = np.where(profile >= half)[0]
    return (x[inds[-1]] - x[inds[0]]) if inds.size >= 2 else 0.0

def main():
    # parameters
    M, pitch = 16, 0.5  # elements, mm
    b = pitch/2; f = 5.0; c = 1500.0

    # grids
    z_fmc = np.linspace(5, 60, 1024)
    x_im  = np.linspace(-8, 8, 201)
    z_im  = np.linspace(5, 60, 271)

    # scatterer
    scatterer_z, scatterer_amp = 40.0, 1.0

    # compute FMC + scatterer
    tof_us, FMC = compute_fmc_with_scatterer(
        M, pitch, b, f, c, z_fmc, scatterer_z, scatterer_amp
    )

    # raw TFM back‐projection
    raw_img = tfm_backprojection(
        M, pitch, x_im, z_im, tof_us, FMC, c
    )
    raw_img /= np.max(np.abs(raw_img))

    # **new**: envelope of the 2D raw image (along depth axis)
    env_img = np.abs(hilbert(raw_img, axis=0))
    env_img /= env_img.max()

    # measure FWHM on envelope
    iz0    = np.argmin(np.abs(z_im - scatterer_z))
    lateral_profile = env_img[iz0, :]
    fwhm_x = measure_fwhm(x_im, lateral_profile)

    ix0    = np.argmin(np.abs(x_im - 0.0))
    axial_profile   = env_img[:, ix0]
    fwhm_z = measure_fwhm(z_im, axial_profile)

    print(f"Envelope‐based lateral FWHM: {fwhm_x:.2f} mm")
    print(f"Envelope‐based axial   FWHM: {fwhm_z:.2f} mm")

    # plot side-by-side
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
    ax2.set_title("Envelope of Raw Reconstruction")
    ax2.set_xlabel("Lateral (mm)")
    ax2.set_ylabel("Depth (mm)")
    plt.colorbar(im2, ax=ax2, label="Normalized envelope")

    # mark scatterer on envelope
    ax2.scatter(0, scatterer_z, c='w', marker='+')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
