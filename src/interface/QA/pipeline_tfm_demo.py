#!/usr/bin/env python3
"""
pipeline_tfm_demo.py

Full-Matrix Capture + Total Focusing Method demo (interface layer).

1) Compute FMC by firing each element and recording all A-scans.
2) Discretize a 2D imaging grid.
3) Delay-and-sum across all Tx-Rx pairs for every grid point.
4) Display and save the TFM image.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

from application.rs_2Dv_service import run_rs_2Dv_service

def compute_fmc(M, pitch, b, f, c, z_mm):
    """
    Build the FMC (M×M×Nz) array of A-scans.
    Returns:
      tof_us : 1D time axis (µs) common to all A-scans
      FMC    : ndarray, shape (M, M, Nz), complex A-scans
    """
    # Element positions (mm)
    x_elem = (np.arange(M) - (M-1)/2) * pitch

    # time‐axis for a reference on‐axis A‐scan
    tof_us = 2 * z_mm * 1e-3 / c * 1e6  # round‐trip time

    FMC = np.zeros((M, M, z_mm.size), dtype=np.complex128)

    for tx in range(M):
        # transmit from element tx at x_elem[tx]
        for rx in range(M):
            # receive on element rx at x_elem[rx]
            # simulate A-scan for this Tx-Rx pair (offset = x_tx → x_rx, but point‐source at 0)
            # we model the same A-scan for any rx, since run_rs returns on-axis; 
            # in true practice you'd shift source/receiver, but for demo we'll re‐use run_rs_2Dv
            p_z = run_rs_2Dv_service(b, f, c, x_elem[tx], 0.0, z_mm)
            # but apply reception delay by shifting p_z in time by rx offset
            # approximate by delaying by extra lateral travel from pixel to rx:
            # here we omit that for simplicity and just record the same p_z
            FMC[tx, rx, :] = p_z

    return tof_us, FMC, x_elem

def run_tfm(FMC, tof_us, x_elem, x_img, z_img, c):
    """
    Perform delay‐and‐sum TFM:
      Image(i,j) = sum_{tx,rx} FMC[tx,rx] sampled at t = (t_tx + t_rx)
    """
    M, _, Nz = FMC.shape
    Nx = x_img.size
    Nz_img = z_img.size

    img = np.zeros((Nz_img, Nx), dtype=np.float64)

    # precompute element positions
    for ix, xi in enumerate(x_img):
        for iz, zi in enumerate(z_img):
            # distance Tx→pixel and pixel→Rx
            d_tx = np.sqrt((xi - x_elem)**2 + zi**2)
            d_rx = d_tx  # same geometry Tx↔Rx
            t_tot = (d_tx + d_rx) * 1e-3 / c * 1e6  # µs

            # sum over all pairs
            s = 0.0
            for tx in range(M):
                for rx in range(M):
                    # find A‐scan sample for this pair (assuming FMC was recorded on tof_us)
                    # use real envelope
                    amp = np.interp(t_tot[tx], tof_us, np.abs(FMC[tx, rx, :]), left=0, right=0)
                    s += amp
            img[iz, ix] = s

    # normalize
    img /= img.max()
    return img

def main():
    parser = argparse.ArgumentParser(
        description="FMC + TFM demo pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--M",    type=int,   default=32,   help="Number of elements")
    parser.add_argument("--pitch",type=float, default=0.5,  help="Element pitch [mm]")
    parser.add_argument("--b",    type=float, default=0.25, help="Half-element size [mm]")
    parser.add_argument("--f",    type=float, default=5.0,  help="Frequency [MHz]")
    parser.add_argument("--c",    type=float, default=1500.0,help="Sound speed [m/s]")

    parser.add_argument("--zmin", type=float, default=1.0,   help="Image z-range start [mm]")
    parser.add_argument("--zmax", type=float, default=60.0,  help="Image z-range end [mm]")
    parser.add_argument("--nx",   type=int,   default=200,   help="Image lateral points")

    args = parser.parse_args()

    # 1) define axial sampling for A-scans
    z_mm = np.linspace(args.zmin, args.zmax, 512)

    # 2) compute FMC
    print("Computing FMC…")
    tof_us, FMC, x_elem = compute_fmc(
        args.M, args.pitch, args.b, args.f, args.c, z_mm
    )

    # 3) define imaging grid
    x_img = np.linspace(-args.M*args.pitch/2, args.M*args.pitch/2, args.nx)
    z_img = np.linspace(args.zmin, args.zmax, args.nx)

    # 4) run TFM
    print("Running TFM (delay-and-sum)…")
    img = run_tfm(FMC, tof_us, x_elem, x_img, z_img, args.c)

    # 5) plot
    plt.figure(figsize=(6, 6))
    plt.imshow(img, extent=[x_img.min(), x_img.max(), z_img.max(), z_img.min()],
               cmap="viridis", aspect='auto')
    plt.xlabel("Lateral (mm)")
    plt.ylabel("Depth (mm)")
    plt.title("TFM Image (normalized)")
    cbar = plt.colorbar(label="Normalized amplitude")
    plt.tight_layout()
    plt.show()

    # 6) save
    out_png = "tfm_image.png"
    plt.imsave(out_png, img, cmap="viridis",
               origin="upper",
               vmin=0, vmax=1)
    print(f"TFM image saved to {out_png}")

if __name__ == "__main__":
    main()
