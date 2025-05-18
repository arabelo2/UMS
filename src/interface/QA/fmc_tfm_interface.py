#!/usr/bin/env python3
"""
Unified FMC → TFM Demo
----------------------
• Generates synthetic FMC data with either a 2‑D line‑source kernel
  (ls2Dv) or a 3‑D rectangular‑piston kernel (ps3Dv).
• Injects a Hann‑windowed N‑cycle tone‑burst echo from a point reflector.
• Forms a classic delay‑and‑sum TFM image.

Example
-------
python interface/fmc_tfm_interface.py \
       --model ps3Dv --lx 0.6 --ly 0.6 \
       --f 5 --c 1480 --xref 0 --zref 40 --plot Y
"""

import argparse, math, os, sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from interface.cli_utils import safe_float, parse_array
from application.ls_2Dv_service import run_ls_2Dv_service
from application.ps_3Dv_service import run_ps_3Dv_service

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser("Synthetic FMC + TFM demo")
parser.add_argument("--model",  choices=["ls2Dv", "ps3Dv"], default="ls2Dv")
parser.add_argument("--M",      type=int,        default=32)
parser.add_argument("--f",      type=safe_float, default=5)      # MHz
parser.add_argument("--c",      type=safe_float, default=1480)   # m/s
parser.add_argument("--dl",     type=safe_float, default=0.5)    # elem length / λ
parser.add_argument("--gd",     type=safe_float, default=0.1)    # gap / length
parser.add_argument("--b",      type=safe_float)                 # 2‑D element half‑length (mm)
parser.add_argument("--lx",     type=safe_float)                 # 3‑D piston dims (mm)
parser.add_argument("--ly",     type=safe_float)
# point reflector
parser.add_argument("--xref",   type=safe_float, default=0)
parser.add_argument("--zref",   type=safe_float, default=40)
# sampling
parser.add_argument("--fs",     type=safe_float, default=20)     # MHz
parser.add_argument("--duration", type=safe_float, default=100)  # µs
parser.add_argument("--burst",  type=int,        default=3)      # cycles
# image grid & plotting
parser.add_argument("--roi_x",  type=parse_array, default="-20,20,401")
parser.add_argument("--roi_z",  type=parse_array, default="5,80,401")
parser.add_argument("--plot",   choices=["Y", "N"], default="Y")
args = parser.parse_args()

# ---------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------
lam      = args.c / (args.f * 1e6)                            # wavelength [m]
pitch_mm = (args.dl * lam * 1e3) + (args.gd * args.dl * lam * 1e3)
centroids_mm = (np.arange(args.M) - (args.M - 1) / 2) * pitch_mm

# ---------------------------------------------------------------------
# time axis
# ---------------------------------------------------------------------
fs_Hz = args.fs * 1e6
nt    = int(round(args.duration * 1e-6 * fs_Hz))
t_sec = np.arange(nt) / fs_Hz                                 # <-- fixed indent

# ---------------------------------------------------------------------
# tone‑burst helper
# ---------------------------------------------------------------------
def tone_burst(cycles: int) -> np.ndarray:
    """Return a complex Hann‑windowed tone burst with <cycles> cycles."""
    ns = int(round(cycles * fs_Hz / (args.f * 1e6)))
    win = 0.5 * (1 - np.cos(2 * np.pi * np.arange(ns) / ns))
    t   = np.arange(ns) / fs_Hz
    return win * np.exp(1j * 2 * np.pi * args.f * 1e6 * t)

burst   = tone_burst(args.burst)
nburst  = len(burst)

# ---------------------------------------------------------------------
# element‑field kernel (single frequency)
# ---------------------------------------------------------------------
def elem_field(tx_mm: float, rx_mm: float) -> complex:
    dx = rx_mm - tx_mm
    if args.model == "ls2Dv":
        b_mm = args.b if args.b else args.dl * lam * 1e3 / 2
        return run_ls_2Dv_service(b_mm / 2, args.f, args.c, tx_mm, x=dx, z=0)
    lx = args.lx if args.lx else args.dl * lam * 1e3
    ly = args.ly if args.ly else lx
    return run_ps_3Dv_service(lx, ly, args.f, args.c,
                              ex=tx_mm, ey=0, x=dx, y=0, z=0)

# ---------------------------------------------------------------------
# FMC synthesis
# ---------------------------------------------------------------------
print("Synthesising FMC …")
FMC = np.zeros((args.M, args.M, nt), dtype=complex)

for tx in range(args.M):
    for rx in range(args.M):
        # monochromatic propagation kernel → analytic time signal
        amp = elem_field(centroids_mm[tx], centroids_mm[rx])      # keep phase
        sig = np.zeros(nt, dtype=complex)                       # empty time trace

        # inject tone‑burst echo from the point reflector
        d_tx = math.hypot(args.xref - centroids_mm[tx], args.zref) * 1e-3
        d_rx = math.hypot(args.xref - centroids_mm[rx], args.zref) * 1e-3
        t_c  = (d_tx + d_rx) / args.c
        k0   = int(round(t_c * fs_Hz)) - nburst // 2
        if 0 <= k0 < nt - nburst:
            sig[k0:k0 + nburst] += amp * burst
        FMC[tx, rx] = sig

# ---------------------------------------------------------------------
# Delay‑and‑sum TFM
# ---------------------------------------------------------------------
print("Running TFM …")
xx, zz = np.meshgrid(args.roi_x, args.roi_z)
img = np.zeros_like(xx, dtype=complex)

for tx in range(args.M):
    for rx in range(args.M):
        tof = (np.hypot(xx - centroids_mm[tx], zz) +
               np.hypot(xx - centroids_mm[rx], zz)) * 1e-3 / args.c
        r_re = np.interp(tof, t_sec, FMC[tx, rx].real, left=0, right=0)
        r_im = np.interp(tof, t_sec, FMC[tx, rx].imag, left=0, right=0)
        img += r_re + 1j * r_im

# ---------------------------------------------------------------------
# Save & plot
# ---------------------------------------------------------------------
np.savez("tfm_image.npz", x=args.roi_x, z=args.roi_z, img=img)

if args.plot.upper() == "Y":
    mag  = np.abs(img)
    peak = np.nanmax(mag) if np.nanmax(mag) > 0 else 1e-12
    with np.errstate(divide="ignore"):
        db = 20 * np.log10(mag / peak)

    plt.figure(figsize=(7, 5))
    pcm = plt.pcolormesh(args.roi_x, args.roi_z, db,
                         shading="auto", cmap="jet",
                         vmin=-40, vmax=0)
    plt.gca().invert_yaxis()
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x [mm]"); plt.ylabel("z [mm]")
    plt.title(f"TFM ({args.model})  f={args.f} MHz  M={args.M}")
    plt.colorbar(pcm, label="Amplitude [dB]")
    plt.tight_layout()
    plt.show()
