#!/usr/bin/env python3
"""
pipeline_scatterer_fwhm_demo.py

Introduce a point scatterer into FMC data, reconstruct with TFM back-projection,
measure axial and lateral FWHM of the scatterer peak, and display the results.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

# Ensure application layer on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from application.delay_laws2D_service import run_delay_laws2D_service
from application.rs_2Dv_service import run_rs_2Dv_service


def compute_fmc_with_scatterer(M, pitch, b, f, c, z_mm, scatterer_z, scatterer_amp):
    """
    Compute FMC dataset for a point scatterer at (x=0, z=scatterer_z).
    Returns tof_us (1D), FMC (M x M x N).
    """
    # 1) time-of-flight grid
    tof_us = 2 * z_mm * 1e-3 / c * 1e6  # µs
    N = len(z_mm)

    # 2) initialize FMC
    FMC = np.zeros((M, M, N))

    # 3) fill FMC via RS_2Dv for each tx-rx pair
    for tx in range(M):
        # compute transmit delay law (unused here for FMC directly)
        td_tx = run_delay_laws2D_service(M, pitch, 0.0, np.inf, c)
        for rx in range(M):
            # receive element offset for TX->RX propagation
            # here we simply record monostatic response (same RS_2Dv) per element
            # element lateral offset
            for m in [tx, rx]:
                pass
            # impulse response at x_eval=0 for element at zero
            # we approximate full matrix by simple monostatic FMC of RS_2Dv(tx) + RS_2Dv(rx)
            # For simplicity, use monostatic simulation per element then sum
            # Actually, generate FMC[tx,rx,:] via two-way RS_2Dv on centered element
            p_z = run_rs_2Dv_service(b, f, c, 0.0, 0.0, z_mm)
            FMC[tx, rx, :] = np.real(p_z)

    # 4) add point scatterer: delay for tx->scatterer->rx
    x_sc = 0.0  # lateral coordinate
    for tx in range(M):
        x_tx = (tx - (M-1)/2) * pitch
        for rx in range(M):
            x_rx = (rx - (M-1)/2) * pitch
            # distance tx->scatterer + scatterer->rx [mm]
            d_tx = np.sqrt((x_sc - x_tx)**2 + scatterer_z**2)
            d_rx = np.sqrt((x_sc - x_rx)**2 + scatterer_z**2)
            t_sc_us = (d_tx + d_rx) * 1e-3 / c * 1e6
            # find nearest sample index
            idx = np.argmin(np.abs(tof_us - t_sc_us))
            FMC[tx, rx, idx] += scatterer_amp

    return tof_us, FMC


def tfm_backprojection(M, pitch, x_im, z_im, tof_us, FMC, c):
    """
    Standard TFM back-projection of FMC dataset.
    """
    Nx = len(x_im)
    Nz = len(z_im)
    img = np.zeros((Nz, Nx))

    for i, xi in enumerate(x_im):
        for j, zj in enumerate(z_im):
            sum_val = 0.0
            for tx in range(M):
                x_tx = (tx - (M-1)/2) * pitch
                d_tx = np.sqrt((xi - x_tx)**2 + zj**2)
                for rx in range(M):
                    x_rx = (rx - (M-1)/2) * pitch
                    d_rx = np.sqrt((xi - x_rx)**2 + zj**2)
                    t_us = (d_tx + d_rx) * 1e-3 / c * 1e6
                    # interpolate FMC
                    sum_val += np.interp(t_us, tof_us, FMC[tx, rx, :], left=0.0, right=0.0)
            img[j, i] = sum_val
    return img


def measure_fwhm(x, profile):
    """
    Measure full-width at half-max of a 1D profile.
    Returns width in same units as x.
    """
    xmax = x[np.argmax(profile)]
    half = np.max(profile) / 2.0
    # find where profile crosses half
    inds = np.where(profile >= half)[0]
    if len(inds) < 2:
        return 0.0
    i1, i2 = inds[0], inds[-1]
    return x[i2] - x[i1]


def main():
    # --- parameters ---
    M = 16
    pitch = 0.5  # mm
    b = pitch/2
    f = 5.0      # MHz
    c = 1500.0   # m/s

    # imaging grid
    x_im = np.linspace(-8, 8, 201)
    z_im = np.linspace(5, 60, 271)

    # scatterer
    scatterer_z =  thirty_z = 40.0  # mm
    scatterer_amp = 1.0

    # depth samples for FMC
    z_fmc = np.linspace(5, 60, 1024)

    # compute FMC with scatterer
    tof_us, FMC = compute_fmc_with_scatterer(M, pitch, b, f, c, z_fmc,
                                            scatterer_z, scatterer_amp)

    # reconstruct TFM image
    img = tfm_backprojection(M, pitch, x_im, z_im, tof_us, FMC, c)
    norm_img = img / np.max(img)

    # measure FWHM at scatterer depth
    # find nearest depth index
    j0 = np.argmin(np.abs(z_im - scatterer_z))
    lateral_profile = norm_img[j0, :]
    fwhm_x = measure_fwhm(x_im, lateral_profile)

    # axial profile at x=0
    i0 = np.argmin(np.abs(x_im - 0.0))
    axial_profile = norm_img[:, i0]
    fwhm_z = measure_fwhm(z_im, axial_profile)

    print(f"Measured lateral FWHM: {fwhm_x:.2f} mm")
    print(f"Measured axial   FWHM: {fwhm_z:.2f} mm")

    # plot
    fig, ax = plt.subplots(1,1,figsize=(6,8))
    im = ax.imshow(norm_img, extent=[x_im[0], x_im[-1], z_im[-1], z_im[0]],
                   cmap='jet', aspect='auto')
    ax.scatter(0, scatterer_z, c='w', marker='+')
    ax.set_xlabel('Lateral (mm)')
    ax.set_ylabel('Depth (mm)')
    ax.set_title('TFM Reconstruction with Point Scatterer')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized amplitude')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
