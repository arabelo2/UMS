#!/usr/bin/env python3
"""
pipeline_fmc_demo.py

Compute full matrix capture (FMC) using delay_laws2D + RS_2Dv (application layer),
save it to disk, and do a quick sanity‑check plot of the summed A-scan.

Usage:
  python pipeline_fmc_demo.py \
    --M 16 --pitch 0.5 --f 5.0 --c 1500.0 \
    --zmin 5 --zmax 60 --nz 512 \
    --out fmc.npy
"""
import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt

# make sure we can import your application layer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from application.delay_laws2D_service import run_delay_laws2D_service
from application.rs_2Dv_service       import run_rs_2Dv_service

def compute_fmc(M: int,
                pitch: float,
                b: float,
                f: float,
                c: float,
                z_mm: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Returns:
      tof_us   : 1D array, the common time‑axis [µs]
      FMC      : 3D array, shape (M, M, len(z_mm)), the FMC matrix
    """
    # 1) get one‑way transmit delays for each element [µs]
    #    here we assume no steering/focus (Φ=0, F=∞) so we get flat delays
    td_us = run_delay_laws2D_service(M, pitch, 0.0, float('inf'), c)

    # time‑of‑flight for each depth sample (two‑way) [µs]
    tof_us = 2 * (z_mm * 1e-3) / c * 1e6

    FMC = np.zeros((M, M, len(z_mm)), dtype=float)

    # 2) fill FMC matrix
    for tx in range(M):
        e_tx = (tx - (M-1)/2) * pitch
        for rx in range(M):
            e_rx = (rx - (M-1)/2) * pitch

            # pressure field for a transmit element at e_tx,
            # observed at lateral position x=e_rx, over all z
            p_z = run_rs_2Dv_service(b, f, c, e_tx, e_rx, z_mm)

            # ensure a 1D real A-scan
            p_z = np.real(p_z)
            p_z = np.atleast_1d(p_z).ravel()

            # build shifted time‑base: tof + tx‑delay + rx‑delay
            xp = tof_us + td_us[tx] + td_us[rx]

            # interpolate onto the common tof_us grid
            FMC[tx, rx, :] = np.interp(tof_us, xp, p_z, left=0.0, right=0.0)

    return tof_us, FMC

def main():
    p = argparse.ArgumentParser(
        description="Compute and save an FMC matrix via delay_laws2D+RS_2Dv."
    )
    p.add_argument("--M",    type=int,   default=16,     help="Num elements")
    p.add_argument("--pitch",type=float, default=0.5,    help="Element pitch [mm]")
    p.add_argument("--f",    type=float, default=5.0,    help="Frequency [MHz]")
    p.add_argument("--c",    type=float, default=1500.0, help="Speed [m/s]")
    p.add_argument("--zmin", type=float, default=5.0,    help="Min depth [mm]")
    p.add_argument("--zmax", type=float, default=60.0,   help="Max depth [mm]")
    p.add_argument("--nz",   type=int,   default=512,    help="Number of z samples")
    p.add_argument("--out",  type=str,   default="fmc.npy", help="Output .npy file")
    args = p.parse_args()

    b = args.pitch/2.   # half‑element width
    z_mm = np.linspace(args.zmin, args.zmax, args.nz)

    print("Computing FMC (this may take a moment)…")
    tof_us, FMC = compute_fmc(args.M, args.pitch, b, args.f, args.c, z_mm)

    # save to disk
    np.save(args.out, FMC)
    print(f"Saved FMC matrix ({args.M}×{args.M}×{args.nz}) to '{args.out}'")

    # quick sanity‑check: sum over all Tx/Rx to get a single A-scan
    summed_ascan = FMC.sum(axis=(0,1))
    # compute envelope via analytic signal (Hilbert via FFT)
    summed_env = np.abs(np.fft.ifft(np.fft.fft(summed_ascan)))

    # plot
    plt.figure(figsize=(6,4))
    plt.plot(tof_us, summed_ascan/np.max(np.abs(summed_ascan)), 'b-', label="RF (norm)")
    plt.plot(tof_us, summed_env  /np.max(summed_env),       'r-', label="Env (norm)")
    plt.xlabel("Time (µs)")
    plt.ylabel("Norm. Amp")
    plt.title("FMC summed A-scan sanity check")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
