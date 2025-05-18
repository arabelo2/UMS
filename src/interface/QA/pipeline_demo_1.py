#!/usr/bin/env python3
"""
pipeline_demo.py

Beamformed A-scan via delay_laws2D + RS_2Dv (application layer),
with a small parameter sweep over steering angle, focus distance, and depth range.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

from application.delay_laws2D_service import run_delay_laws2D_service
from application.rs_2Dv_service import run_rs_2Dv_service

def beamform_ascan(M, pitch_mm, steering_deg, focus_mm, c_m_s, half_elem_mm, freq_mhz, z_mm):
    """
    Compute the beamformed RF and envelope for given parameters.

    Returns:
      tof_us : 1D numpy array of time points (µs)
      rf     : beamformed RF trace (real, same length as tof_us)
      env    : envelope of rf (same length)
    """
    # 1) per‐element delays (µs)
    delays_us = run_delay_laws2D_service(M, pitch_mm, steering_deg, focus_mm, c_m_s)  # shape (M,)

    # 2) time‐of‐flight for each depth sample (µs)
    #    2*z [mm] → meters: *1e-3, /c [m/s] → seconds, *1e6 → µs
    tof_us = 2 * z_mm * 1e-3 / c_m_s * 1e6  # shape (N,)

    # 3) accumulate beamformed RF
    rf = np.zeros_like(tof_us)

    for m in range(M):
        # element lateral offset (mm), centered at zero
        offset_mm = (m - (M - 1) / 2) * pitch_mm

        # compute 1D A-scan at x=0
        p_z = run_rs_2Dv_service(half_elem_mm, freq_mhz, c_m_s, offset_mm, 0.0, z_mm)
        # squeeze away any singleton dims to get a 1D vector
        fp = np.real(p_z).squeeze()

        # shifted time‐axis for this element
        xp = (tof_us + delays_us[m]).squeeze()

        # add into rf (zero‐pad outside xp range)
        rf += np.interp(tof_us, xp, fp, left=0.0, right=0.0)

    # 4) envelope detection (hilbert requires real input)
    env = np.abs(hilbert(rf))

    return tof_us, rf, env

def plot_ascan(tof_us, rf, env, title):
    """Plot the RF and normalized envelope for one A-scan."""
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    ax1.plot(tof_us, rf, 'b-')
    ax1.set_ylabel("Beamformed RF", fontsize=12)
    ax1.grid(True)

    ax2.plot(tof_us, env / np.max(env), 'r-')
    ax2.set_xlabel("Time (µs)", fontsize=12)
    ax2.set_ylabel("Normalized Envelope", fontsize=12)
    ax2.grid(True)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    # --- array/transducer parameters ---
    M = 16           # number of elements
    pitch = 0.5      # element pitch [mm]
    half_elem = pitch / 2  # half-element dimension [mm]
    freq = 5.0       # frequency [MHz]
    c = 1500.0       # sound speed [m/s]

    # --- parameter sweeps ---
    phis = [-20, -10, 0, 10, 20]          # steering angles [deg]
    focuses = [np.inf, 100.0, 50.0]       # focal distances [mm]
    depth_ranges = [
        (5.0, 60.0, 512),   # z_min, z_max, n_points
        (10.0, 40.0, 512)
    ]

    for phi in phis:
        for F in focuses:
            for zmin, zmax, npts in depth_ranges:
                z_mm = np.linspace(zmin, zmax, npts)
                tof_us, rf, env = beamform_ascan(
                    M, pitch, phi, F, c, half_elem, freq, z_mm
                )

                title = (
                    f"Φ = {phi}°, F = "
                    + ("∞" if np.isinf(F) else f"{F:.0f} mm")
                    + f", z = [{zmin:.1f}, {zmax:.1f}] mm"
                )
                plot_ascan(tof_us, rf, env, title)
