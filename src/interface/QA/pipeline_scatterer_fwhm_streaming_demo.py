#!/usr/bin/env python3
"""
pipeline_scatterer_fwhm_streaming_demo.py

Memory-efficient TFM + scatterer FWHM measurement:
  - caches each element’s A-scan once (O(M) simulator calls)
  - streams through Tx/Rx pairs, injects scatterer, builds image on the fly
  - measures lateral & axial FWHM of the point scatterer
"""
import sys, os
import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt

# bring application layer into path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from application.rs_2Dv_service import run_rs_2Dv_service

def compute_tof_and_cache(M, pitch, b, f, c, z_mm):
    """
    Returns:
      tof_us : (Nz,) common 2-way time axis [µs]
      p_cache: dict[i]→(Nz,) 1D real A-scan for element i
    """
    elem_pos = (np.arange(M) - (M - 1)/2) * pitch
    tof_us   = 2 * z_mm * 1e-3 / c * 1e6

    p_cache = {}
    for i, x_elem in enumerate(elem_pos):
        raw = run_rs_2Dv_service(b, f, c, x_elem, 0.0, z_mm)
        p_cache[i] = np.real(raw).squeeze()
    return tof_us, p_cache

def tfm_scatterer_streaming(M, pitch, x_im, z_im,
                            tof_us, p_cache, c,
                            scatterer_z, scatterer_amp):
    """
    Streams through all Tx/Rx pairs, injects the point scatterer
    at the correct TOF sample, and accumulates the backprojection.
    """
    elem_pos = (np.arange(M) - (M - 1)/2) * pitch
    Nx, Nz   = x_im.size, z_im.size
    img      = np.zeros((Nz, Nx), float)

    # precompute Hilbert envelopes once
    env_cache = {
        i: np.abs(hilbert(p_cache[i], axis=-1))
        for i in p_cache
    }

    x_sc = 0.0  # scatterer lateral position
    # loop over all Tx/Rx pairs
    for tx in range(M):
        for rx in range(M):
            env_sum = env_cache[tx] + env_cache[rx]

            # inject scatterer amplitude at its two-way time
            d_tx_sc = np.hypot(x_sc - elem_pos[tx], scatterer_z) * 1e-3
            d_rx_sc = np.hypot(x_sc - elem_pos[rx], scatterer_z) * 1e-3
            t_sc_us = (d_tx_sc + d_rx_sc) / c * 1e6
            idx = np.argmin(np.abs(tof_us - t_sc_us))
            env_sum[idx] += scatterer_amp

            # project this pair onto the image grid
            for ix, x in enumerate(x_im):
                d_tx = np.hypot(x - elem_pos[tx], z_im) * 1e-3
                d_rx = np.hypot(x - elem_pos[rx], z_im) * 1e-3
                t_us = (d_tx + d_rx) / c * 1e6
                img[:, ix] += np.interp(t_us, tof_us, env_sum,
                                        left=0.0, right=0.0)
    # normalize
    return img / img.max()

def measure_fwhm(x, profile):
    """
    Measure full-width at half-max of a 1D profile.
    """
    half = np.max(profile) / 2.0
    inds = np.where(profile >= half)[0]
    return (x[inds[-1]] - x[inds[0]]) if inds.size >= 2 else 0.0

def main():
    # --- parameters ---
    M, pitch = 16, 0.5  # elements, mm
    b = pitch/2; f = 5.0; c = 1500.0

    # sampling grids
    z_fmc = np.linspace(5, 60, 1024)
    x_im  = np.linspace(-8, 8, 201)
    z_im  = np.linspace(5, 60, 271)

    # scatterer specs
    scatterer_z, scatterer_amp = 40.0, 1.0

    # cache A-scans & time axis
    tof_us, p_cache = compute_tof_and_cache(M, pitch, b, f, c, z_fmc)

    # reconstruct + inject scatterer
    img = tfm_scatterer_streaming(
        M, pitch, x_im, z_im,
        tof_us, p_cache, c,
        scatterer_z, scatterer_amp
    )

    # measure FWHM
    iz0 = np.argmin(np.abs(z_im - scatterer_z))
    fwhm_x = measure_fwhm(x_im, img[iz0, :])
    ix0 = np.argmin(np.abs(x_im - 0.0))
    fwhm_z = measure_fwhm(z_im, img[:, ix0])

    print(f"Measured lateral FWHM: {fwhm_x:.2f} mm")
    print(f"Measured axial   FWHM: {fwhm_z:.2f} mm")

    # display image
    plt.figure(figsize=(6, 8))
    plt.imshow(img, extent=[x_im[0], x_im[-1], z_im[-1], z_im[0]],
               aspect='auto', cmap='jet')
    plt.scatter(0, scatterer_z, c='w', marker='+')
    plt.xlabel("Lateral (mm)")
    plt.ylabel("Depth (mm)")
    plt.title("TFM + Point Scatterer")
    plt.colorbar(label="Normalized amplitude")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
