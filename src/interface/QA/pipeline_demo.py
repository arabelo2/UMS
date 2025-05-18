#!/usr/bin/env python3
"""
pipeline_demo.py

Demonstration of a simple beamforming pipeline using application-layer services:
  - delay_laws2D_service.run_delay_laws2D_service
  - rs_2Dv_service.run_rs_2Dv_service

This script:
  1. Computes per-element transmission delays (µs) for a 1-D array.
  2. Simulates the received A-scan on each element via Rayleigh–Sommerfeld 2D.
  3. Applies those delays, sums across elements to form an RF A-scan.
  4. Extracts and normalizes the envelope via the analytic signal.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

from application.delay_laws2D_service import run_delay_laws2D_service
from application.rs_2Dv_service import run_rs_2Dv_service

def pipeline_demo():
    # —— Parameters —— #
    M = 16                    # number of elements
    pitch = 0.5               # element pitch (mm)
    f = 5.0                   # frequency (MHz)
    c = 1540.0                # wave speed (m/s)
    steering_angle = 0.0      # steering angle (degrees)
    focus_F = np.inf          # focal distance (mm): np.inf = steering only

    # Depth axis for A-scan (mm)
    z_axis = np.linspace(10, 50, 1024)

    # Single lateral scan position (mm)
    scan_x = 0.0

    # Compute per-element delays (µs)
    delays_us = run_delay_laws2D_service(M, pitch, steering_angle, focus_F, c)

    # Convert depths to two-way time-of-flight (µs)
    tof_us = 2.0 * (z_axis * 1e-3) / c * 1e6

    # Element lateral positions (mm)
    elem_x = (np.arange(M) - (M - 1) / 2) * pitch

    # Accumulate RF signal
    rf = np.zeros_like(tof_us)

    for m in range(M):
        # Simulate received complex pressure for element m
        p_complex = run_rs_2Dv_service(
            pitch/2,      # half-element width (mm)
            f, c,
            elem_x[m],    # element lateral offset
            scan_x,       # evaluation x
            z_axis        # evaluation depths
        )

        # Collapse to a 1D real magnitude vector
        p_mag = np.abs(p_complex).squeeze()

        # Build the delayed time axis for this element
        delayed_t = tof_us + delays_us[m]

        # Interpolate onto the common tof_us grid
        rf += np.interp(tof_us, delayed_t, p_mag, left=0.0, right=0.0)

    # Extract and normalize envelope
    envelope = np.abs(hilbert(rf))
    envelope /= np.max(envelope)

    # Plot the result
    plt.figure(figsize=(10, 4))
    plt.plot(tof_us, envelope, lw=1.5)
    plt.xlabel("Time (µs)")
    plt.ylabel("Normalized Envelope")
    plt.title("Beamformed A-scan via delay_laws2D + RS_2Dv (app layer)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    pipeline_demo()
